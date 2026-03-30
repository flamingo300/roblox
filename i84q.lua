--!optimize 2

local max, min, huge = math.max, math.min, math.huge

---- Declarations ----

export type UDim          = { Scale: number, Offset: number }
export type UDim2         = { X: UDim, Y: UDim }

export type PointInstance = {
    Destroy: (self: PointInstance) -> (),
    CFrame: CFrame, Size: Vector3, Active: boolean
}

export type Point3D = {
    Destroy: (self: Point3D) -> (),
    Position: Vector3, Active: boolean
}

export type PointModel = {
    Destroy: (self: PointModel) -> (),
    CFrame: CFrame, Size: Vector3, Active: boolean
}

type Point = PointInstance | Point3D | PointModel

export type Cluster = {
    Destroy: (self: Cluster) -> (),
    Pause:   (self: Cluster) -> (),
    Resume:  (self: Cluster) -> ()
}

---- Constants ----

local RunService = game:GetService 'RunService'

Vector2.zero = Vector2.new()

local Weak = table.freeze { __mode = 'k' }

local CornerSigns = table.freeze {
    table.freeze {-1, -1, -1},
    table.freeze { 1, -1, -1},
    table.freeze {-1,  1, -1},
    table.freeze { 1,  1, -1},
    table.freeze {-1, -1,  1},
    table.freeze { 1, -1,  1},
    table.freeze {-1,  1,  1},
    table.freeze { 1,  1,  1},
}

local ModelPointCache = setmetatable({}, Weak)

---- Classes ----

local UDim2 = {}; do
    --- Constructor ---

    function UDim2.new(xScale: number, xOffset: number, yScale: number, yOffset: number): UDim2
        return setmetatable(
            {
                X = { Scale = xScale, Offset = xOffset },
                Y = { Scale = yScale, Offset = yOffset }
            },

            UDim2
        ) :: any
    end

    function UDim2.fromScale(xScale: number, yScale: number): UDim2
        return UDim2.new(xScale, 0, yScale, 0)
    end

    function UDim2.fromOffset(xOffset: number, yOffset: number): UDim2
        return UDim2.new(0, xOffset, 0, yOffset)
    end

    --- Metatables ---

    UDim2.__index = UDim2
end

local DefaultSize     = UDim2.fromScale(1, 1)
local DefaultPosition = UDim2.fromScale(0, 0)
local DefaultAnchor   = Vector2.new(0, 0)

local Point3D = {}; do
    --- Constructor ---

    function Point3D.new(position: Vector3): Point3D
        assert(typeof(position) == 'vector', `invalid argument #1 to 'Point3D.new': expected Vector3, got {type(position)}`)

        -- Exports --
        return setmetatable(
            {
                Position = position,
                Active   = true
            },

            Point3D
        ) :: any
    end

    --- Methods ---

    function Point3D:Destroy(): ()
        self.Active, self.Position = false, nil
    end

    --- Metatables ---

    Point3D.__index = Point3D
end

local PointInstance = {}; do
    --- Constructor ---

    function PointInstance.new(instance: BasePart): PointInstance
        assert(typeof(instance) == 'Instance', `invalid argument #1 to 'PointInstance.new': expected Instance, got {typeof(instance)}`)

        -- Exports --
        return setmetatable(
            {
                Instance = instance,
                Active   = true
            },

            PointInstance
        ) :: any
    end

    --- Methods ---

    function PointInstance:Destroy(): ()
        self.Active, self.Instance = false, nil
    end

    --- Metatables ---

    function PointInstance:__index(key: string): any
        local inst = rawget(self, 'Instance')

        if key == 'CFrame' then
            return inst and inst.CFrame or nil
        elseif key == 'Size' then
            return inst and inst.Size or nil
        end

        return PointInstance[key]
    end
end

local PointModel = {}; do
    --- Constructor ---

    function PointModel.new(instance: Instance): PointModel
        assert(typeof(instance) == 'Instance', `invalid argument #1 to 'PointModel.new': expected Instance, got {typeof(instance)}`)
        assert(instance:IsA('Model') or instance:IsA('Folder'), `invalid argument #1 to 'PointModel.new': expected Model or Folder, got {instance.ClassName}`)

        local pts = {}
        local n = 0

        local follow: BasePart? = nil

        for _, d in instance:GetDescendants() do
            if d:IsA('BasePart') then
                follow = follow or d

                local cf = d.CFrame
                local half = d.Size / 2

                local pos = cf.Position
                local r = cf.RightVector
                local u = cf.UpVector
                local l = cf.LookVector

                local hx, hy, hz = half.X, half.Y, half.Z

                for _, s in CornerSigns do
                    local lx, ly, lz = s[1] * hx, s[2] * hy, s[3] * hz
                    n += 1
                    pts[n] = pos + r * lx + u * ly + l * lz
                end
            end
        end

        if instance:IsA('Model') then
            local pp = instance.PrimaryPart
            if pp then
                follow = pp
            end
        end

        if n == 0 then
            return setmetatable(
                {
                    Instance = instance,
                    Active   = true,
                    Follow   = follow,

                    _RelPos  = nil,
                    _RelR    = nil,
                    _RelU    = nil,
                    _RelL    = nil,
                    _Size    = Vector3.zero
                },

                PointModel
            ) :: any
        end

        -- Centroid --
        local sx, sy, sz = 0, 0, 0
        for i = 1, n do
            local p = pts[i]
            sx += p.X; sy += p.Y; sz += p.Z
        end
        local invN = 1 / n
        local mx, my, mz = sx * invN, sy * invN, sz * invN

        -- Covariance matrix (symmetric, upper triangle) --
        local cxx, cxy, cxz = 0, 0, 0
        local cyy, cyz = 0, 0
        local czz = 0

        for i = 1, n do
            local p = pts[i]
            local x = p.X - mx
            local y = p.Y - my
            local z = p.Z - mz

            cxx += x * x
            cxy += x * y
            cxz += x * z
            cyy += y * y
            cyz += y * z
            czz += z * z
        end

        cxx *= invN; cxy *= invN; cxz *= invN
        cyy *= invN; cyz *= invN
        czz *= invN

        -- Power iteration for dominant eigenvector --
        local function normalize(x: number, y: number, z: number): (number, number, number)
            local m = (x * x + y * y + z * z) ^ 0.5
            if m == 0 then return 1, 0, 0 end
            local inv = 1 / m
            return x * inv, y * inv, z * inv
        end

        local vx, vy, vz = 1, 0, 0
        for _ = 1, 8 do
            local nx = cxx * vx + cxy * vy + cxz * vz
            local ny = cxy * vx + cyy * vy + cyz * vz
            local nz = cxz * vx + cyz * vy + czz * vz
            vx, vy, vz = normalize(nx, ny, nz)
        end

        -- Orthonormal basis from dominant eigenvector --
        local ax, ay, az = 0, 1, 0
        local dot = vx * ax + vy * ay + vz * az
        if (dot < 0 and -dot or dot) > 0.9 then
            ax, ay, az = 0, 0, 1
        end

        local ux, uy, uz = normalize(
            vy * az - vz * ay,
            vz * ax - vx * az,
            vx * ay - vy * ax
        )

        local wx, wy, wz = normalize(
            vy * uz - vz * uy,
            vz * ux - vx * uz,
            vx * uy - vy * ux
        )

        -- Project all points onto OBB axes --
        local min1, min2, min3 = huge, huge, huge
        local max1, max2, max3 = -huge, -huge, -huge

        for i = 1, n do
            local p = pts[i]
            local x = p.X - mx
            local y = p.Y - my
            local z = p.Z - mz

            local d1 = x * vx + y * vy + z * vz
            local d2 = x * ux + y * uy + z * uz
            local d3 = x * wx + y * wy + z * wz

            min1 = min(min1, d1); max1 = max(max1, d1)
            min2 = min(min2, d2); max2 = max(max2, d2)
            min3 = min(min3, d3); max3 = max(max3, d3)
        end

        -- OBB center and size --
        local cx = (min1 + max1) * 0.5
        local cy = (min2 + max2) * 0.5
        local cz = (min3 + max3) * 0.5

        local size = Vector3.new(max1 - min1, max2 - min2, max3 - min3)

        local worldCenter = Vector3.new(
            mx + vx * cx + ux * cy + wx * cz,
            my + vy * cx + uy * cy + wy * cz,
            mz + vz * cx + uz * cy + wz * cz
        )

        -- Store relative transform against Follow part --
        local relPos = worldCenter
        local rv = Vector3.new(vx, vy, vz)
        local uv = Vector3.new(ux, uy, uz)
        local lv = Vector3.new(wx, wy, wz)

        local relR = rv
        local relU = uv
        local relL = lv

        if follow then
            local fcf = follow.CFrame
            local fpos = fcf.Position
            local fr = fcf.RightVector
            local fu = fcf.UpVector
            local fl = fcf.LookVector

            local dx = worldCenter - fpos
            relPos = Vector3.new(vector.dot(dx, fr), vector.dot(dx, fu), vector.dot(dx, fl))

            relR = Vector3.new(vector.dot(rv, fr), vector.dot(rv, fu), vector.dot(rv, fl))
            relU = Vector3.new(vector.dot(uv, fr), vector.dot(uv, fu), vector.dot(uv, fl))
            relL = Vector3.new(vector.dot(lv, fr), vector.dot(lv, fu), vector.dot(lv, fl))
        end

        -- Exports --
        return setmetatable(
            {
                Instance = instance,
                Active   = true,
                Follow   = follow,

                _RelPos  = relPos,
                _RelR    = relR,
                _RelU    = relU,
                _RelL    = relL,
                _Size    = size
            },

            PointModel
        ) :: any
    end

    --- Methods ---

    function PointModel:Destroy(): ()
        self.Active, self.Instance, self.Follow, self._RelPos, self._RelR, self._RelU, self._RelL, self._Size =
            false, nil, nil, nil, nil, nil, nil, nil
    end

    --- Metatables ---

    function PointModel:__index(key: string): any
        local inst = rawget(self, 'Instance')

        if key == 'CFrame' then
            if not inst then return nil end

            local follow = rawget(self, 'Follow')
            local relPos = rawget(self, '_RelPos')
            local relR   = rawget(self, '_RelR')
            local relU   = rawget(self, '_RelU')
            local relL   = rawget(self, '_RelL')

            if not (relPos and relR and relU and relL) then
                return nil
            end

            if follow then
                local fcf  = follow.CFrame
                local fpos = fcf.Position
                local fr   = fcf.RightVector
                local fu   = fcf.UpVector
                local fl   = fcf.LookVector

                local wCenter =
                    fpos
                    + fr * relPos.X
                    + fu * relPos.Y
                    + fl * relPos.Z

                local wR =
                    fr * relR.X
                    + fu * relR.Y
                    + fl * relR.Z

                local wU =
                    fr * relU.X
                    + fu * relU.Y
                    + fl * relU.Z

                local wL =
                    fr * relL.X
                    + fu * relL.Y
                    + fl * relL.Z

                return CFrame.fromMatrix(wCenter, wR, wU, wL)
            end

            return CFrame.fromMatrix(relPos, relR, relU, relL)
        elseif key == 'Size' then
            if not inst then return nil end
            return rawget(self, '_Size')
        end

        return PointModel[key]
    end
end

local Prototype = {}; do
    --- Metatables ---

    Prototype.__index = Prototype

    --- Methods ---

    function Prototype:Pause(): ()
        self.Paused = true
    end

    function Prototype:Resume(): ()
        self.Paused = false
    end

    function Prototype:Destroy(): ()
        self.Active = false

        local conn = self.Connection
        if conn then
            conn:Disconnect()
            self.Connection = nil
        end

        for drawObj in self.Attachments do
            drawObj:Remove()
        end

        table.clear(self.Attachments)
    end
end

---- Functions ----

@native
local function project(cf: CFrame, size: Vector3, camera: Camera): Vector2
    local half = size / 2

    local pos = cf.Position
    local r   = cf.RightVector
    local u   = cf.UpVector
    local l   = cf.LookVector

    local hx, hy, hz = half.X, half.Y, half.Z

    local minX, minY = huge, huge
    local maxX, maxY = -huge, -huge

    local projected = false
    for _, s in CornerSigns do
        local lx, ly, lz = s[1] * hx, s[2] * hy, s[3] * hz
        local w = pos + r * lx + u * ly + l * lz

        local scr, vis = camera:WorldToScreenPoint(w)
        if vis then
            projected = true

            local sx, sy = scr.X, scr.Y
            minX = min(minX, sx); minY = min(minY, sy)
            maxX = max(maxX, sx); maxY = max(maxY, sy)
        end
    end

    if not projected then return Vector2.zero end
    return Vector2.new(max(0, maxX - minX), max(0, maxY - minY))
end

@native
local function screen(udim: UDim, reference: number): number
    return udim.Scale * reference + udim.Offset
end

---- Drawing ----

function Drawing.attach(descriptor: { [any]: {
    Link:        Point?,
    From:        Point?,
    To:          Point?,
    Size:        UDim2?,
    Position:    UDim2?,
    AnchorPoint: Vector2?
} }): Cluster
    local attachments = {}

    local cluster = setmetatable(
        {
            Attachments = attachments,
            Active      = true,
            Paused      = false,
            Connection  = nil
        },

        Prototype
    )

    for object, config in descriptor do
        assert(config.Link or (config.From and config.To), `Drawing.attach: 'Link', or 'From' & 'To' are required.`)

        attachments[object] = {
            Link        = config.Link,
            From        = config.From,
            To          = config.To,
            Size        = config.Size or DefaultSize,
            Position    = config.Position or DefaultPosition,
            AnchorPoint = config.AnchorPoint or DefaultAnchor
        }
    end

    @native
    local function Update()
        if not cluster.Active or cluster.Paused then return end

        local currentCamera = workspace.CurrentCamera
        local viewportSize  = currentCamera.ViewportSize
        local cleanup       = true

        for drawObj, attach in attachments do
            local link = attach.Link
            local from = attach.From
            local to   = attach.To

            if typeof(link) == 'Instance' then
                if link:IsA('Model') or link:IsA('Folder') then
                    local wrapped = ModelPointCache[link]
                    if not wrapped then
                        wrapped = PointModel.new(link)
                        ModelPointCache[link] = wrapped
                    end
                    link = wrapped
                    attach.Link = wrapped
                end
            end

            local isLine    = from ~= nil and to ~= nil
            local destroyed = false

            if link and typeof(link) == 'table' and not link.Active then
                destroyed = true
            end

            if not destroyed then
                local inst = if typeof(link) == 'table' then rawget(link, 'Instance') else link
                if inst and typeof(inst) == 'Instance' and not inst.Parent then
                    destroyed = true
                end
            end

            if isLine then
                if not from.Active then destroyed = true end
                if not to.Active   then destroyed = true end
            end

            if destroyed then
                drawObj.Visible = false
                continue
            end

            cleanup = false

            if isLine then
                local wFrom: Vector3? = if from.CFrame then from.CFrame.Position elseif from.Position then from.Position else nil
                local wTo: Vector3?   = if to.CFrame   then to.CFrame.Position   elseif to.Position   then to.Position   else nil

                if not wFrom or not wTo then
                    drawObj.Visible = false
                    continue
                end

                local sFrom, fromVisible = currentCamera:WorldToScreenPoint(wFrom)
                local sTo,   toVisible   = currentCamera:WorldToScreenPoint(wTo)

                if not fromVisible or not toVisible then
                    drawObj.Visible = false
                    continue
                end

                drawObj.From    = sFrom
                drawObj.To      = sTo
                drawObj.Visible = true
            else
                local wPos: Vector3? = nil

                if typeof(link) == 'table' then
                    wPos = if link.CFrame then link.CFrame.Position elseif link.Position then link.Position else nil
                elseif typeof(link) == 'Instance' then
                    if link:IsA('BasePart') then
                        wPos = link.Position
                    end
                end

                if not wPos then
                    drawObj.Visible = false
                    continue
                end

                local sPos, visible = currentCamera:WorldToScreenPoint(wPos)

                if not visible then
                    drawObj.Visible = false
                    continue
                end

                local projectedSize = Vector2.zero

                local inst = if typeof(link) == 'table' then rawget(link, 'Instance') else link
                if inst and typeof(inst) == 'Instance' and inst:IsA('BasePart') then
                    projectedSize = project(inst.CFrame, inst.Size, currentCamera)
                elseif typeof(link) == 'table' then
                    local cf = link.CFrame
                    local sz = link.Size
                    if cf and sz then
                        projectedSize = project(cf, sz, currentCamera)
                    end
                end

                local attSize = attach.Size
                local attPos  = attach.Position
                local anchor  = attach.AnchorPoint

                local width  = screen(attSize.X, projectedSize.X)
                local height = screen(attSize.Y, projectedSize.Y)

                drawObj.Size     = vector.ceil(vector.create(width, height))
                drawObj.Position = vector.ceil(vector.create(
                    (sPos.X + screen(attPos.X, viewportSize.X)) - (width  * anchor.X),
                    (sPos.Y + screen(attPos.Y, viewportSize.Y)) - (height * anchor.Y)
                ))

                drawObj.Visible = true
            end
        end

        if cleanup then
            cluster:Destroy()
        end
    end

    cluster.Connection = RunService.Render:Connect(Update)

    return cluster :: any
end

---- Exports ----

_G.UDim2         = UDim2
_G.PointInstance  = PointInstance
_G.Point3D       = Point3D
_G.PointModel    = PointModel

---- definitions ----

type EnumItem = {
    EnumType: Enum,
    Value: number,
    Name: string
}

type Enum = {
    items: { [string]: EnumItem },
    name: string,

    insert: (string, number) -> Enum,

    GetEnumItems: () -> { [string]: EnumItem },
    FromName: (string) -> EnumItem?,
    FromValue: (number) -> EnumItem?,
}

---- classes ----

local EnumItem = {}; do
    --- constructor ---

    local function constructor(
        name: string,
        value: number,
        parent: Enum
    ): EnumItem
        assert('string' == type(name),                   `bad argument #1 to EnumItem.new: string expected, got '{type(name)}'`)
        assert('number' == type(value),                  `bad argument #2 to EnumItem.new: number expected, got '{type(value)}'`)
        assert('table' == type(parent) and parent.items, `bad argument #3 to EnumItem.new: Enum expected, got '{type(parent)}'`)

        local self = setmetatable( {
            EnumType = parent,
            Value = value,
            Name = name
        }, EnumItem )

        -- insertion --

        parent.items[name] = self

        -- exports --

        return self
    end

    --- functions ---

    EnumItem.new = constructor

    --- metatables ---

    function EnumItem:__tostring()
        return `Enum.{self.EnumType.name}.{self.Name}`
    end

    EnumItem.__index = EnumItem
end

local Enums = setmetatable( {
    GetEnums = function(self)
        return self.items
    end,

    items = {}
}, {
    __index = function(self, index)
        return self.items[index]
    end
} )

local Enum = {}; do
    --- constructor ---

    local function constructor(name: string)
        assert('string' == type(name), `bad argument #1 to Enum.new: string expected, got '{type(name)}'`)

        local self = setmetatable( {
            items = {},
            name = name
        }, Enum )

        -- insertion --

        Enums.items[name] = self

        -- exports --

        return self
    end

    --- methods ---

    function Enum:GetEnumItems()
        return self.items
    end

    function Enum:FromName(name: string)
        for _, Item: EnumItem in self.items do
            if Item.Name == name then
                return Item
            end
        end

        return nil
    end

    function Enum:FromValue(value: number)
        for _, Item: EnumItem in self.items do
            if Item.Value == value then
                return Item
            end
        end

        return nil
    end

    function Enum:insert(name: string, value: number)
        EnumItem.new(name, value, self)

        return self
    end

    --- functions ---

    Enum.new = constructor

    --- metatables ---

    function Enum:__tostring()
        return self.name
    end

    function Enum:__index(key)
        return self.items and self.items[key] or rawget(Enum, key)
    end
end

---- runtime ----

local API
local time = os.clock()

local function regenerate()
    local content = crypt.json.decode(game:HttpGet('https://raw.githubusercontent.com/MaximumADHD/Roblox-Client-Tracker/refs/heads/roblox/Full-API-Dump.json'))

    content.time = time
    writefile('api.bin', crypt.base64.encode(crypt.json.encode(content)))

    return content
end

if not isfile('api.bin') then
    API = regenerate()
else
    local content = crypt.json.decode(crypt.base64.decode(readfile('api.bin')))

    if content.time < time - 259200 then
        API = regenerate()
    else
        API = content
    end
end

for index, data in API.Enums do
    local enum = Enum.new(data.Name)

    for _, item in data.Items do
        enum:insert(item.Name, item.Value)
    end
end

Instance.declare {
    class = 'Instance',
    name  = 'IsA',
    callback = {
        method = function(self, className)
            local currentClass = self.ClassName

            if currentClass == className then
                return true
            end

            for _, classData in API.Classes do
                if classData.Name == currentClass then
                    local superclass = classData.Superclass

                    while superclass and superclass ~= "<<<ROOT>>>" do
                        if superclass == className then
                            return true
                        end

                        local found = false
                        for _, parentClassData in API.Classes do
                            if parentClassData.Name == superclass then
                                superclass = parentClassData.Superclass
                                found = true
                                break
                            end
                        end

                        if not found then
                            break
                        end
                    end

                    break
                end
            end

            return false
        end
    }
}

---- exports ----

_G.Enum = table.freeze(Enums)

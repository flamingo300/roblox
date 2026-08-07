---- assignments ----

_G.bit64         = require('@types/integers/bit64');
_G.fs            = require('@libraries/fs');
_G.http          = require('@libraries/http');
_G.signal        = require('@external/signal');

_G.UDim2         = require('@types/primitives/udim2');
_G.Point3D       = require('@types/pointers/world');
_G.Point2D       = require('@types/pointers/screen');
_G.PointInstance = require('@types/pointers/instance');
_G.PointModel    = require('@types/pointers/model');

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

---- drawing ----

local render = require('@routines/render');

(_G.Drawing :: any).attach = render.attach;

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

---- declarations ----

Instance.declare {
    class    = 'Instance',
    name     = 'Read',
    callback = function(self: Instance, offset: number, size: number): buffer
        return memory.readbuffer(self, offset or 0, size or 32);
    end;
};

Instance.declare {
    class    = 'Instance',
    name     = 'Write',
    callback = function(self: Instance, offset: number, data: buffer): ()
        return memory.writebuffer(self, offset or 0, data);
    end;
};

---- runtime ----

task.delay(1, function()
    local cache: string? = nil do
        if fs.file('spec.d.txt') then
            local ok, content = pcall(fs.read, 'spec.d.txt');
            if ok and type(content) == 'string' then
                cache = content:match('^%s*(%S+)%s*$');
            end;
        end;
    end;

    local remote: string? = nil do
        local response = http.get { url = 'https://api.github.com/repos/flamingo300/roblox/contents/luau/spec.d.luau' };

        if type(response) == 'string' and #response > 0 then
            local ok, data = pcall(crypt.json.decode, response);

            if ok and type(data) == 'table' and type(data.sha) == 'string' then
                remote = data.sha;
            end;
        end;
    end;

    if remote and cache and remote == cache and fs.file('spec.d.luau') then
        return;
    end;

    if not remote and fs.file('spec.d.luau') then
        return;
    end;

    local response = http.get { url = 'https://raw.githubusercontent.com/flamingo300/roblox/master/luau/spec.d.luau' };
    if type(response) == 'string' and #response > 0 then
        fs.open('spec.d.luau')
            :write(response)
            :close();

        if remote then
            pcall(fs.write, 'spec.d.txt', remote);
        end;
    end;
end);

local API
local time = os.clock()

local function regenerate()
    local content = crypt.json.decode(game:HttpGet('https://raw.githubusercontent.com/MaximumADHD/Roblox-Client-Tracker/refs/heads/roblox/Full-API-Dump.json'))

    content.time = time
    fs.write('api.bin', crypt.base64.encode(crypt.json.encode(content)))

    return content
end

if not fs.file('api.bin') then
    API = regenerate()
else
    local content = crypt.json.decode(crypt.base64.decode(fs.read('api.bin')))

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

--!optimize 2

---- definitions ----

export type PointModel = {
    Instance: Instance?,
    Follow:   Instance?,
    Active:   boolean,
    Destroy:  (self: PointModel) -> (),
    CFrame:   CFrame?,
    Size:     vector?,

    _RelPos:  vector?,
    _RelR:    vector?,
    _RelU:    vector?,
    _RelL:    vector?,
    _Size:    vector?,
};

---- constants ----

local huge        = math.huge;
local max         = math.max;
local min         = math.min;

local create      = vector.create;
local dot         = vector.dot;

local CornerSigns = table.freeze({
    table.freeze({ -1, -1, -1 }),
    table.freeze({  1, -1, -1 }),
    table.freeze({ -1,  1, -1 }),
    table.freeze({  1,  1, -1 }),
    table.freeze({ -1, -1,  1 }),
    table.freeze({  1, -1,  1 }),
    table.freeze({ -1,  1,  1 }),
    table.freeze({  1,  1,  1 }),
} :: any);

---- module ----

local PointModel = {}; do
    --- constructors ---

    function PointModel.new(instance: Instance): PointModel
        assert(typeof(instance) == "Instance", `invalid argument #1 to 'PointModel.new': expected Instance, got {typeof(instance)}`);
        assert(instance:IsA("Model") or instance:IsA("Folder"), `invalid argument #1 to 'PointModel.new': expected Model or Folder, got {instance.ClassName}`);

        local follow: Instance?, points: { vector }, n: number = nil, {}, 0;
        for _, d in instance:GetDescendants() do
            if (d:IsA("BasePart")) then
                follow = follow or d;

                local cframe: CFrame = (d :: any).CFrame;
                local size:   vector = (d :: any).Size;
                local half:   vector = size * 0.5;

                local
                    position,
                    right_vector,
                    up_vector,
                    look_vector
                = cframe.Position, cframe.RightVector, cframe.UpVector, cframe.LookVector;

                local
                    hx: number,
                    hy: number,
                    hz: number
                = half.x, half.y, half.z;

                for _, s in CornerSigns do
                    local
                        lx: number,
                        ly: number,
                        lz: number
                    = (s :: any)[1] * hx, (s :: any)[2] * hy, (s :: any)[3] * hz;

                    n += 1;
                    points[n] = position + right_vector * lx + up_vector * ly + look_vector * lz;
                end;
            end;
        end;

        if (instance:IsA("Model")) then
            follow = if(follow) then instance.PrimaryPart else follow;
        end;

        if (n == 0) then
            return setmetatable(
                {
                    Instance = instance,
                    Active   = true,
                    Follow   = follow,

                    _RelPos  = nil,
                    _RelR    = nil,
                    _RelU    = nil,
                    _RelL    = nil,
                    _Size    = vector.zero,
                },

                PointModel
            ) :: any;
        end;

        local sx: number = 0;
        local sy: number = 0;
        local sz: number = 0;

        for i = 1, n do
            local p: vector = points[i];
            sx += p.x;
            sy += p.y;
            sz += p.z;
        end;

        local inv_n: number = 1 / n;
        local mx:   number  = sx * inv_n;
        local my:   number  = sy * inv_n;
        local mz:   number  = sz * inv_n;

        local cxx: number = 0;
        local cxy: number = 0;
        local cxz: number = 0;
        local cyy: number = 0;
        local cyz: number = 0;
        local czz: number = 0;

        for i = 1, n do
            local p: vector = points[i];
            local x: number = p.x - mx;
            local y: number = p.y - my;
            local z: number = p.z - mz;

            cxx += x * x;
            cxy += x * y;
            cxz += x * z;
            cyy += y * y;
            cyz += y * z;
            czz += z * z;
        end;

        cxx *= inv_n;
        cxy *= inv_n;
        cxz *= inv_n;
        cyy *= inv_n;
        cyz *= inv_n;
        czz *= inv_n;

        local function normalize(x: number, y: number, z: number): (number, number, number)
            local m: number = (x * x + y * y + z * z) ^ 0.5;
            if (m == 0) then
                return 1, 0, 0;
            end;
            local inv: number = 1 / m;
            return x * inv, y * inv, z * inv;
        end;

        local vx: number = 1;
        local vy: number = 0;
        local vz: number = 0;

        for _ = 1, 8 do
            local nx: number = cxx * vx + cxy * vy + cxz * vz;
            local ny: number = cxy * vx + cyy * vy + cyz * vz;
            local nz: number = cxz * vx + cyz * vy + czz * vz;
            vx, vy, vz = normalize(nx, ny, nz);
        end;

        local ax:  number = 0;
        local ay:  number = 1;
        local az:  number = 0;
        local dot: number = vx * ax + vy * ay + vz * az;

        if ((if (dot < 0) then -dot else dot) > 0.9) then
            ax, ay, az = 0, 0, 1;
        end;

        local ux: number, uy: number, uz: number = normalize(
            vy * az - vz * ay,
            vz * ax - vx * az,
            vx * ay - vy * ax
        );

        local wx: number, wy: number, wz: number = normalize(
            vy * uz - vz * uy,
            vz * ux - vx * uz,
            vx * uy - vy * ux
        );

        local min1: number = huge;
        local min2: number = huge;
        local min3: number = huge;
        local max1: number = -huge;
        local max2: number = -huge;
        local max3: number = -huge;

        for i = 1, n do
            local p: vector = points[i];
            local x: number = p.x - mx;
            local y: number = p.y - my;
            local z: number = p.z - mz;

            local d1: number = x * vx + y * vy + z * vz;
            local d2: number = x * ux + y * uy + z * uz;
            local d3: number = x * wx + y * wy + z * wz;

            min1 = min(min1, d1);
            max1 = max(max1, d1);
            min2 = min(min2, d2);
            max2 = max(max2, d2);
            min3 = min(min3, d3);
            max3 = max(max3, d3);
        end;

        local cx: number = (min1 + max1) * 0.5;
        local cy: number = (min2 + max2) * 0.5;
        local cz: number = (min3 + max3) * 0.5;

        local size: vector = create(max1 - min1, max2 - min2, max3 - min3);

        local world_center: vector = create(
            mx + vx * cx + ux * cy + wx * cz,
            my + vy * cx + uy * cy + wy * cz,
            mz + vz * cx + uz * cy + wz * cz
        );

        local rel_pos: vector = world_center;
        local rel_r:   vector = create(vx, vy, vz);
        local rel_u:   vector = create(ux, uy, uz);
        local rel_l:   vector = create(wx, wy, wz);

        if (follow) then
            local fcf:  CFrame = (follow :: any).CFrame;
            local fpos: vector = fcf.Position;
            local fr:   vector = fcf.RightVector;
            local fu:   vector = fcf.UpVector;
            local fl:   vector = fcf.LookVector;

            local dx: vector = world_center - fpos;
            rel_pos = create(dot(dx, fr), dot(dx, fu), dot(dx, fl));

            local rv: vector = create(vx, vy, vz);
            local uv: vector = create(ux, uy, uz);
            local lv: vector = create(wx, wy, wz);

            rel_r = create(dot(rv, fr), dot(rv, fu), dot(rv, fl));
            rel_u = create(dot(uv, fr), dot(uv, fu), dot(uv, fl));
            rel_l = create(dot(lv, fr), dot(lv, fu), dot(lv, fl));
        end;

        return setmetatable(
            {
                Instance = instance,
                Active   = true,
                Follow   = follow,

                _RelPos  = rel_pos,
                _RelR    = rel_r,
                _RelU    = rel_u,
                _RelL    = rel_l,
                _Size    = size,
            },

            PointModel
        ) :: any;
    end;

    --- methods ---

    function PointModel.Destroy(self: PointModel): ()
        self.Active   = false;
        (self :: any).Instance = nil;
        (self :: any).Follow   = nil;
        (self :: any)._RelPos  = nil;
        (self :: any)._RelR    = nil;
        (self :: any)._RelU    = nil;
        (self :: any)._RelL    = nil;
        (self :: any)._Size    = nil;
    end;

    --- metatables ---

    function PointModel.__index(self: PointModel, key: string): any
        local inst = rawget(self, "Instance" :: any);

        if (key == "CFrame") then
            if (not inst) then
                return nil;
            end;

            local follow  = rawget(self, "Follow" :: any);
            local rel_pos = rawget(self, "_RelPos" :: any);
            local rel_r   = rawget(self, "_RelR" :: any);
            local rel_u   = rawget(self, "_RelU" :: any);
            local rel_l   = rawget(self, "_RelL" :: any);

            if (not (rel_pos and rel_r and rel_u and rel_l)) then
                return nil;
            end;

            if (follow) then
                local fcf:  CFrame  = (follow :: any).CFrame;
                local fpos: vector = fcf.Position;
                local fr:   vector   = fcf.RightVector;
                local fu:   vector   = fcf.UpVector;
                local fl:   vector   = fcf.LookVector;

                local w_center: vector =
                    fpos
                    + fr * (rel_pos :: vector).x
                    + fu * (rel_pos :: vector).y
                    + fl * (rel_pos :: vector).z;

                local w_r: vector =
                    fr * (rel_r :: vector).x
                    + fu * (rel_r :: vector).y
                    + fl * (rel_r :: vector).z;

                local w_u: vector =
                    fr * (rel_u :: vector).x
                    + fu * (rel_u :: vector).y
                    + fl * (rel_u :: vector).z;

                local w_l: vector =
                    fr * (rel_l :: vector).x
                    + fu * (rel_l :: vector).y
                    + fl * (rel_l :: vector).z;

                return CFrame.fromMatrix(w_center, w_r, w_u, w_l);
            end;

            return CFrame.fromMatrix(rel_pos :: vector, rel_r :: vector, rel_u :: vector, rel_l :: vector);
        elseif (key == "Size") then
            if (not inst) then
                return nil;
            end;
            return rawget(self, "_Size" :: any);
        end;

        return (PointModel :: any)[key];
    end;
end;

---- exports ----

return PointModel;

--!optimize 2

---- environment ----

local vector = vector;
local table  = table;
local math   = math;
local game   = game;

local create = vector.create;
local ceil   = vector.ceil;
local zero   = vector.zero;

local huge   = math.huge;
local max    = math.max;
local min    = math.min;

---- definitions ----

export type UDim = {
    Scale:  number,
    Offset: number,
};

export type UDim2 = {
    X: UDim,
    Y: UDim,
};

export type Point3D = {
    Position: vector,
    Active:   boolean,
    Destroy:  (self: Point3D) -> (),
};

export type Point2D = {
    Position: vector,
    Active:   boolean,
    Destroy:  (self: Point2D) -> (),
};

export type PointInstance = {
    Instance: Instance?,
    Active:   boolean,
    CFrame:   CFrame?,
    Size:     vector?,
};

export type PointModel = {
    Instance: Instance?,
    Follow:   Instance?,
    Active:   boolean,
    CFrame:   CFrame?,
    Size:     vector?,
};

export type Point = Point3D | Point2D | PointInstance | PointModel | Instance;

export type Attachment = {
    Link:        Point?,
    From:        Point?,
    To:          Point?,
    Size:        UDim2,
    Position:    UDim2,
    AnchorPoint: vector,
};

export type Cluster = {
    Attachments: { [any]: Attachment },
    Active:      boolean,
    Paused:      boolean,
    Connection:  any,

    Pause:   (self: Cluster) -> (),
    Resume:  (self: Cluster) -> (),
    Destroy: (self: Cluster) -> (),
};

export type Drawing_module = {
    attach: (descriptor: { [any]: { Link: Point?, From: Point?, To: Point?, Size: UDim2?, Position: UDim2?, AnchorPoint: vector? } }) -> Cluster,
};

---- constants ----

local UDim2       = require("@types/primitives/udim2");
local PointModel  = require("@types/pointers/model");
local Point2D_mod = require("@types/pointers/screen");

local RunService = game:GetService("RunService");

local Weak = table.freeze({ __mode = "k" } :: any);

local CornerSigns = table.freeze({
    table.freeze({ -1, -1, -1 } :: any),
    table.freeze({  1, -1, -1 } :: any),
    table.freeze({ -1,  1, -1 } :: any),
    table.freeze({  1,  1, -1 } :: any),
    table.freeze({ -1, -1,  1 } :: any),
    table.freeze({  1, -1,  1 } :: any),
    table.freeze({ -1,  1,  1 } :: any),
    table.freeze({  1,  1,  1 } :: any),
} :: any);

local ModelPointCache = setmetatable({}, Weak);

local DefaultSize     = UDim2.fromScale(1, 1);
local DefaultPosition = UDim2.fromScale(0, 0);
local DefaultAnchor   = create(0, 0);

---- classes ----

local Prototype = {} :: Cluster; do
    Prototype.__index = Prototype;

    function Prototype.Pause(self: Cluster): ()
        self.Paused = true;
    end;

    function Prototype.Resume(self: Cluster): ()
        self.Paused = false;
    end;

    function Prototype.Destroy(self: Cluster): ()
        self.Active = false;

        local conn = rawget(self, "Connection" :: any);
        if (conn) then
            (conn :: any):Disconnect();
            rawset(self, "Connection" :: any, nil);
        end;

        local atts = rawget(self, "Attachments" :: any);
        if (atts) then
            for draw_obj, _ in atts do
                (draw_obj :: any):Remove();
            end;
            table.clear(atts :: any);
        end;
    end;
end;

---- functions ----

@native
local function project(cf: CFrame, size: vector, camera: any): vector
    local half: vector = size * 0.5;

    local pos: vector = cf.Position;
    local r:   vector = cf.RightVector;
    local u:   vector = cf.UpVector;
    local l:   vector = cf.LookVector;

    local hx: number = half.x;
    local hy: number = half.y;
    local hz: number = half.z;

    local min_x: number = huge;
    local min_y: number = huge;
    local max_x: number = -huge;
    local max_y: number = -huge;

    local projected: boolean = false;

    for _, s in CornerSigns do
        local sx: number = (s :: any)[1];
        local sy: number = (s :: any)[2];
        local sz: number = (s :: any)[3];

        local lx: number = sx * hx;
        local ly: number = sy * hy;
        local lz: number = sz * hz;

        local w: vector = pos + r * lx + u * ly + l * lz;

        local scr: vector, vis: boolean = (camera :: any):WorldToScreenPoint(w);
        if (vis) then
            projected = true;

            local x: number = scr.x;
            local y: number = scr.y;

            if (x < min_x) then min_x = x; end;
            if (y < min_y) then min_y = y; end;
            if (x > max_x) then max_x = x; end;
            if (y > max_y) then max_y = y; end;
        end;
    end;

    if (not projected) then
        return zero;
    end;

    return create(max(0, max_x - min_x), max(0, max_y - min_y));
end;

@native
local function screen(udim: UDim, reference: number): number
    return udim.Scale * reference + udim.Offset;
end;

---- module ----

local Drawing = {} :: Drawing_module; do
    function Drawing.attach(descriptor: { [any]: { Link: Point?, From: Point?, To: Point?, Size: UDim2?, Position: UDim2?, AnchorPoint: vector? } }): Cluster
        assert(type(descriptor) == "table", `Drawing.attach: expected table, got {type(descriptor)}`);

        local attachments: { [any]: Attachment } = {};

        local cluster = setmetatable(
            {
                Attachments = attachments,
                Active      = true,
                Paused      = false,
                Connection  = nil,
            },

            Prototype
        ) :: any;

        for object, config in descriptor do
            assert(type(config) == "table", `Drawing.attach: expected config table, got {type(config)}`);
            assert(config.Link or (config.From and config.To), `Drawing.attach: 'Link', or 'From' & 'To' are required.`);

            attachments[object] = {
                Link        = config.Link,
                From        = config.From,
                To          = config.To,
                Size        = config.Size or DefaultSize,
                Position    = config.Position or DefaultPosition,
                AnchorPoint = config.AnchorPoint or DefaultAnchor,
            };
        end;

        @native
        local function Update(): ()
            if (not cluster.Active or cluster.Paused) then
                return;
            end;

            local current_camera: any = (workspace :: any).CurrentCamera;
            if (not current_camera) then
                return;
            end;

            local viewport_size: vector = (current_camera :: any).ViewportSize;
            local cleanup: boolean = true;

            for draw_obj, attach in attachments do
                local link: any = attach.Link;
                local from: any = attach.From;
                local to:   any = attach.To;

                --- lazy wrap Model/Folder Instance -> PointModel ---

                if (typeof(link) == "Instance") then
                    if ((link :: Instance):IsA("Model") or (link :: Instance):IsA("Folder")) then
                        local cached = ModelPointCache[link];
                        if (not cached) then
                            cached = PointModel.new(link :: Instance);
                            ModelPointCache[link] = cached;
                        end;
                        link = cached;
                        attach.Link = cached;
                    end;
                end;

                local is_line: boolean = from ~= nil and to ~= nil;
                local destroyed: boolean = false;

                if (type(link) == "table" and not (link :: any).Active) then
                    destroyed = true;
                end;

                if (not destroyed) then
                    local inst: any = if (type(link) == "table") then rawget(link :: any, "Instance") else link;
                    if (inst and typeof(inst) == "Instance" and not (inst :: Instance).Parent) then
                        destroyed = true;
                    end;
                end;

                if (is_line) then
                    if (type(from) == "table" and not (from :: any).Active) then
                        destroyed = true;
                    end;
                    if (type(to) == "table" and not (to :: any).Active) then
                        destroyed = true;
                    end;
                end;

                if (destroyed) then
                    (draw_obj :: any).Visible = false;
                    continue;
                end;

                cleanup = false;

                if (is_line) then
                    local w_from: vector? = nil;
                    local w_to:   vector? = nil;

                    local from_mt = if (type(from) == "table") then getmetatable(from :: any) else nil;
                    local to_mt   = if (type(to) == "table") then getmetatable(to :: any) else nil;

                    local from_is_screen: boolean = from_mt == Point2D_mod;
                    local to_is_screen:   boolean = to_mt == Point2D_mod;

                    if (from_is_screen) then
                        w_from = (from :: any).Position;
                    elseif (type(from) == "table") then
                        local cf: any = (from :: any).CFrame;
                        local ps: any = (from :: any).Position;
                        w_from = if (cf) then (cf :: CFrame).Position elseif (ps) then (ps :: vector) else nil;
                    elseif (typeof(from) == "Instance") then
                        if ((from :: Instance):IsA("BasePart")) then
                            w_from = (from :: any).Position;
                        end;
                    end;

                    if (to_is_screen) then
                        w_to = (to :: any).Position;
                    elseif (type(to) == "table") then
                        local cf: any = (to :: any).CFrame;
                        local ps: any = (to :: any).Position;
                        w_to = if (cf) then (cf :: CFrame).Position elseif (ps) then (ps :: vector) else nil;
                    elseif (typeof(to) == "Instance") then
                        if ((to :: Instance):IsA("BasePart")) then
                            w_to = (to :: any).Position;
                        end;
                    end;

                    if (not w_from or not w_to) then
                        (draw_obj :: any).Visible = false;
                        continue;
                    end;

                    local s_from: vector;
                    local s_to:   vector;
                    local from_vis: boolean;
                    local to_vis:   boolean;

                    if (from_is_screen) then
                        s_from = w_from :: vector;
                        from_vis = true;
                    else
                        s_from, from_vis = (current_camera :: any):WorldToScreenPoint(w_from :: vector);
                    end;

                    if (to_is_screen) then
                        s_to = w_to :: vector;
                        to_vis = true;
                    else
                        s_to, to_vis = (current_camera :: any):WorldToScreenPoint(w_to :: vector);
                    end;

                    if (not from_vis or not to_vis) then
                        (draw_obj :: any).Visible = false;
                        continue;
                    end;

                    (draw_obj :: any).From    = s_from;
                    (draw_obj :: any).To      = s_to;
                    (draw_obj :: any).Visible = true;
                else
                    local link_is_screen: boolean = if (type(link) == "table") then getmetatable(link :: any) == Point2D_mod else false;

                    local w_pos: vector? = nil;
                    local s_pos: vector? = nil;
                    local visible: boolean = false;

                    if (link_is_screen) then
                        w_pos = (link :: any).Position;
                        s_pos = w_pos;
                        visible = true;
                    elseif (type(link) == "table") then
                        local cf: any = (link :: any).CFrame;
                        local ps: any = (link :: any).Position;
                        w_pos = if (cf) then (cf :: CFrame).Position elseif (ps) then (ps :: vector) else nil;
                        if (w_pos) then
                            s_pos, visible = (current_camera :: any):WorldToScreenPoint(w_pos :: vector);
                        end;
                    elseif (typeof(link) == "Instance") then
                        if ((link :: Instance):IsA("BasePart")) then
                            w_pos = (link :: any).Position;
                            s_pos, visible = (current_camera :: any):WorldToScreenPoint(w_pos :: vector);
                        end;
                    end;

                    if (not w_pos or not s_pos or not visible) then
                        (draw_obj :: any).Visible = false;
                        continue;
                    end;

                    local projected_size: vector = zero;

                    local inst: any = if (type(link) == "table") then rawget(link :: any, "Instance") else link;
                    if (inst and typeof(inst) == "Instance" and (inst :: Instance):IsA("BasePart")) then
                        projected_size = project((inst :: any).CFrame, (inst :: any).Size, current_camera);
                    elseif (type(link) == "table") then
                        local cf: any = (link :: any).CFrame;
                        local sz: any = (link :: any).Size;
                        if (cf and sz) then
                            projected_size = project(cf :: CFrame, sz :: vector, current_camera);
                        elseif (link_is_screen) then
                            projected_size = zero;
                        end;
                    end;

                    local att_size: UDim2 = attach.Size;
                    local att_pos:  UDim2 = attach.Position;
                    local anchor:   vector = attach.AnchorPoint;

                    local width:  number = screen(att_size.X, projected_size.x);
                    local height: number = screen(att_size.Y, projected_size.y);

                    (draw_obj :: any).Size     = ceil(create(width, height));
                    (draw_obj :: any).Position = ceil(create(
                        (s_pos :: vector).x + screen(att_pos.X, viewport_size.x) - (width  * anchor.x),
                        (s_pos :: vector).y + screen(att_pos.Y, viewport_size.y) - (height * anchor.y)
                    ));

                    (draw_obj :: any).Visible = true;
                end;
            end;

            if (cleanup) then
                (cluster :: Cluster):Destroy();
            end;
        end;

        cluster.Connection = (RunService :: any).Render:Connect(Update);

        return cluster :: any;
    end;
end;

---- exports ----

return Drawing;

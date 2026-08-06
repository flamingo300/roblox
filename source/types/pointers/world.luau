--!optimize 2

---- definitions ----

export type Point3D = {
    Position: vector,
    Active:   boolean,
    Destroy:  (self: Point3D) -> (),
};

---- module ----

local Point3D = {}; do
    --- constructors ---

    function Point3D.new(position: vector): Point3D
        assert(type(position) == "vector", `invalid argument #1 to 'Point3D.new': expected vector, got {type(position)}`);

        return setmetatable(
            {
                Position = position,
                Active   = true,
            },

            Point3D
        ) :: any;
    end;

    --- methods ---

    function Point3D.Destroy(self: Point3D): ()
        self.Active   = false;
        (self :: any).Position = nil;
    end;

    --- metatables ---

    Point3D.__index = Point3D;
end;

---- exports ----

return Point3D;

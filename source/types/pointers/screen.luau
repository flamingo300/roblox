--!optimize 2

---- definitions ----

export type Point2D = {
    Position: vector,
    Active:   boolean,
    Destroy:  (self: Point2D) -> (),
};

---- module ----

local Point2D = {}; do
    --- constructors ---

    function Point2D.new(position: vector): Point2D
        assert(type(position) == "vector", `invalid argument #1 to 'Point2D.new': expected vector, got {type(position)}`);

        return setmetatable(
            {
                Position = position,
                Active   = true,
            },

            Point2D
        ) :: any;
    end;

    --- methods ---

    function Point2D.Destroy(self: Point2D): ()
        self.Active = false;
        (self :: any).Position = nil;
    end;

    --- metatables ---

    Point2D.__index = Point2D;
end;

---- exports ----

return Point2D;

--!optimize 2

---- definitions ----

export type UDim = {
    Scale:  number,
    Offset: number,
};

export type UDim2 = {
    X: UDim,
    Y: UDim,
};

---- module ----

local UDim2 = {}; do
    --- constructors ---

    function UDim2.new(x0: number, x1: number, y0: number, y1: number): UDim2
        assert(type(x0) == "number",  `invalid argument #1 to 'UDim2.new': expected number, got {type(x0)}`);
        assert(type(x1) == "number", `invalid argument #2 to 'UDim2.new': expected number, got {type(x1)}`);
        assert(type(y0) == "number",  `invalid argument #3 to 'UDim2.new': expected number, got {type(y0)}`);
        assert(type(y1) == "number", `invalid argument #4 to 'UDim2.new': expected number, got {type(y1)}`);

        return setmetatable(
            {
                X = { Scale = x0, Offset = x1 },
                Y = { Scale = y0, Offset = y1 },
            },

            UDim2
        ) :: any;
    end;

    function UDim2.fromScale(x: number, y: number): UDim2
        assert(type(x) == "number", `invalid argument #1 to 'UDim2.fromScale': expected number, got {type(x)}`);
        assert(type(y) == "number", `invalid argument #2 to 'UDim2.fromScale': expected number, got {type(y)}`);

        return UDim2.new(x, 0, y, 0);
    end;

    function UDim2.fromOffset(x: number, y: number): UDim2
        assert(type(x) == "number", `invalid argument #1 to 'UDim2.fromOffset': expected number, got {type(x)}`);
        assert(type(y) == "number", `invalid argument #2 to 'UDim2.fromOffset': expected number, got {type(y)}`);

        return UDim2.new(0, x, 0, y);
    end;

    --- metatables ---

    UDim2.__index = UDim2;
end;

---- exports ----

return UDim2;

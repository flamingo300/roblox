--!optimize 2

---- definitions ----

export type PointInstance = {
    Instance: Instance?,
    Active:   boolean,
    Destroy:  (self: PointInstance) -> (),
    CFrame:   CFrame?,
    Size:     vector?,
};

---- module ----

const PointInstance = {}; do
    --- constructors ---

    function PointInstance.new(instance: Instance): PointInstance
        assert(typeof(instance) == "Instance", `invalid argument #1 to 'PointInstance.new': expected Instance, got {typeof(instance)}`);

        return setmetatable(
            {
                Instance = instance,
                Active   = true,
            },

            PointInstance
        ) :: any;
    end;

    --- methods ---

    function PointInstance.Destroy(self: PointInstance): ()
        self.Active   = false;
        (self :: any).Instance = nil;
    end;

    --- metatables ---

    function PointInstance.__index(self: PointInstance, key: string): any
        const inst = rawget(self, "Instance" :: any);

        if (key == "CFrame") then
            return if (inst) then (inst :: any).CFrame else nil;
        elseif (key == "Size") then
            return if (inst) then (inst :: any).Size else nil;
        end;

        return (PointInstance :: any)[key];
    end;
end;

---- exports ----

return PointInstance;

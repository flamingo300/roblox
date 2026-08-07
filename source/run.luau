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

---- drawing ----

local render = require('@routines/render');

(_G.Drawing :: any).attach = render.attach;

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

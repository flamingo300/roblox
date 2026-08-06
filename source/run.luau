require('@archive/gena')

---- assignments ----

_G.bit64  = require('@types/bit64')
_G.fs     = require('@libraries/fs')
_G.http   = require('@libraries/http')
_G.signal = require('@external/signal')

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

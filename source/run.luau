require('@archive/gena')

---- assignments ----

_G.bit64  = require('@types/bit64')
_G.fs     = require('@libraries/fs')
_G.http   = require('@libraries/http')
_G.signal = require('@external/signal')

---- runtime ----

task.delay(1, function()
    local spec = http.get { url = 'https://github.com/flamingo300/roblox/blob/master/luau/spec.d.luau?raw=true' };

    if(spec) then
        fs.open('spec.d.luau')
            :write(spec)
            :close()
    end;
end);

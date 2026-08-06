--!nolint
--!nocheck

return {
    get = function(args): string
        return game:HttpGet(args.url, args.content);
    end;

    post = function(args): string
        return game:HttpPost(args.url, args.content, args.type, args.accept, args.cookie, args.referrer, args.origin);
    end;
} :: typeof(http);

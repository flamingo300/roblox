return {
    get = function(args: {
        url:      string,
        content:  string
    }): string
        return game:HttpGet(args.url, args.content);
    end;

    post = function(args: {
        url:      string,
        content:  string,
        type:     string,
        accept:   string,
        cookie:   string,
        referrer: string,
        origin:   string
    }): string
        return game:HttpPost(args.url, args.content, args.type, args.accept, args.cookie, args.referrer, args.origin);
    end;
}

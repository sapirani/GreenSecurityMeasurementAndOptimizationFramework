import logging


class PrettyExtrasFormatter(logging.Formatter):
    def format(self, record):
        base = super().format(record)

        # Get non-standard attributes (extras)
        standard_attrs = vars(logging.LogRecord('', '', '', 0, '', (), None)).keys()
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith('_')
        }

        if not extras:
            return base

        # Format extras pretty
        max_key_len = max(len(k) for k in extras)
        lines = [
            f"│ {k.ljust(max_key_len)} : {v}"
            for k, v in sorted(extras.items())
        ]
        box = "\n".join([
            "╭── Extra Fields",
            *lines,
            "╰───────────────────────────────"
        ])
        return f"{base}\n{box}"

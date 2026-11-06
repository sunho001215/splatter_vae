"""Private YAML text rendering for the two legacy registration publishers."""


def render_appended_jobs(before: bytes, rows: list[dict], *, yaml, json) -> str:
    """Append rows at the existing jobs node boundary, retaining legacy layout and errors."""
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    return (text[:node.end_mark.index].rstrip() + '\n'
            + yaml.safe_dump(json.loads(json.dumps(rows)), sort_keys=False) + '\n' + text[node.end_mark.index:])

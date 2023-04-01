def generate_prompt(_input, _output=None):
    if _output is not None:
        return f"""{_input}{_output}"""
    else:
        return f"""{_input}"""
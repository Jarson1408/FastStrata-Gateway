import re

_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)

def extract_tool_calls(content: str):
    calls = []
    cleaned = content
    while True:
        idx_start = cleaned.find("<tool_call>")
        if idx_start == -1:
            break
            
        search_idx = idx_start + 11
        depth = 1
        while depth > 0:
            next_start = cleaned.find("<tool_call>", search_idx)
            next_end = cleaned.find("</tool_call>", search_idx)
            
            if next_end == -1:
                break
                
            if next_start != -1 and next_start < next_end:
                depth += 1
                search_idx = next_start + 11
            else:
                depth -= 1
                search_idx = next_end + 12
                
        if depth > 0:
            break
            
        end_idx = search_idx
        block = cleaned[idx_start:end_idx]
        
        m_func = re.search(r"<function=([^>\s]+)>", block)
        if m_func:
            fn_name = m_func.group(1).strip()
            body_start = m_func.end()
            body_end = block.rfind("</function>")
            if body_end != -1:
                body = block[body_start:body_end]
                
                args = {}
                param_search_idx = 0
                while True:
                    m_param = re.search(r"<parameter=([^>\s]+)>", body[param_search_idx:])
                    if not m_param:
                        break
                    p_name = m_param.group(1).strip()
                    p_start = param_search_idx + m_param.end()
                    
                    p_depth = 1
                    p_idx = p_start
                    while p_depth > 0:
                        n_start = body.find("<parameter=", p_idx)
                        n_end = body.find("</parameter>", p_idx)
                        
                        if n_end == -1:
                            break
                            
                        if n_start != -1 and n_start < n_end:
                            p_depth += 1
                            p_idx = n_start + 11
                        else:
                            p_depth -= 1
                            p_idx = n_end + 12
                            
                    if p_depth > 0:
                        break
                        
                    p_end = p_idx - 12
                    p_val = body[p_start:p_end]
                    args[p_name] = p_val.strip()
                    param_search_idx = p_idx
                    
                calls.append({"name": fn_name, "args": args})
                
        cleaned = cleaned[:idx_start] + cleaned[end_idx:]
        
    return cleaned, calls

test_content = """
Here is the update:
<tool_call>
<function=edit>
<parameter=targetContent>
<tool_call>
<function=NAME>
<parameter=KEY>VALUE</parameter>
</function>
</tool_call>
</parameter>
</function>
</tool_call>
"""
print(extract_tool_calls(test_content))

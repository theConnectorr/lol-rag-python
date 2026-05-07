# src/core/text_utils.py
def flatten_toc(items, current_section=""):
    """Hàm đệ quy làm phẳng mục lục (TOC)"""
    flat_list = []
    for item in items:
        section_name = item.get("title", "")
        text_content = item.get("textContent", "")
        
        if text_content:
            flat_list.append({"text": text_content, "section": section_name})
            
        children = item.get("children", [])
        if children:
            flat_list.extend(flatten_toc(children, section_name))
            
    return flat_list

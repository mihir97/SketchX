from yattag import Doc, indent
import json
data = json.load(open('data.json', 'r'))

def generate_html(data,css):
    doc, tag, text = Doc().tagtext()
    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('style'):
                doc.asis(css)
        with tag('body'):
            for d in data:
                doc.asis(d)

    result = indent(doc.getvalue())
    print(result)

def generate_css(name,data):
    output_str = []
    output_str.append("\n"+name)
    output_str.append(' {\n')
    for key in data:
        output_str.append("\t"+key+": "+data[key]+";\n")
    output_str.append("}\n")
    return ''.join(output_str)

content = []
css_list = []
xmax = data["xmax"]
ymax = data["ymax"]

#TODO: Sort the regions in a top down fashion
for element in data['regions']:
    doc, tag, text = Doc().tagtext()
    if element["type"] == "Label":
        with tag('label'):
            text(element["content"])
        content.append(doc.getvalue())
    elif element["type"] == "TextBox":
        doc.stag('input', type = 'text', value = element['content'])
        content.append(doc.getvalue())

css_list.append( generate_css("body",{"display":"flex","flex-direction":"column"}))
generate_html(content,'\n'.join(css_list))



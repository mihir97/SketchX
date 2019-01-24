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
            with tag('h1'):
                text('Hello world!')

    result = indent(doc.getvalue())
    print(result)

def generate_css(name,data):
    output_str = []
    output_str.append(name)
    output_str.apend(' {\n')
    for key in data:
        output_str.append("\t"+key+": "+data[key]+";\n")
    output_str.append("}\n")
    return ''.join(output_str)

content = []
xmax = data["xmax"]
ymax = data["ymax"]

doc, tag, text = Doc().tagtext()
#TODO: Sort the regions in a top down fashion
for element in data['regions']:
    if element["type"] == "Label":
        with tag('label'):
            text(element["content"])
            content.append(doc.getvalue())

generate_html(content,"")



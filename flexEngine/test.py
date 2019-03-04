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
    with open('test.html', 'w') as HTMLCode:
        HTMLCode.write(result)

def generate_css_util():
    # for element in data['regions']:
        # generate_css(element["type"], {CSS Styles })
        # Calculations will go here
        # Location, Height, Width and other stuff
        # Sibling and parent relations based on len(tag(region))
        # will provide further details about the flex columns or row 
        pass

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
        content.append("<br/>")
    elif element["type"] == "TextBox":
        doc.stag('input', type = 'text', placeholder = element['content'], id = element['id'])
        content.append(doc.getvalue())
        content.append("<br/>")
    elif element["type"] == "CheckBox":
        with tag('input', type = 'checkbox', value = element['content'], id = element['id']):
            text(element["content"])
        content.append(doc.getvalue())
        content.append("<br/>")
    elif element["type"] == "Button":
        doc.stag('input', type = 'button', value = element['content'], id = element['id'])
        content.append(doc.getvalue())
    elif element["type"] == "Image":
        doc.stag('img', src=element['src'], height=element['height'], width= element['width'])
        content.append(doc.getvalue())

css_list.append( generate_css("body",{"display":"flex","flex-direction":"column", 
"margin" : "auto", "margin-top": "15px" ,"width": "60%"}))
# css_list.append(generate_css_util())
generate_html(content,'\n'.join(css_list))

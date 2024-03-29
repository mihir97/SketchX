from yattag import Doc, indent
import random
import json
import sys
data = json.load(open(sys.argv[1], 'r'))
prev_y = 0
prev_h = 0
consecutive_radio = 0

def generate_html(data,css):
    doc, tag, text = Doc().tagtext()
    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            doc.asis('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">')
            doc.asis('<link rel="stylesheet" href="form.css">')
            with tag('script', src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"):
                pass
            with tag('script', src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"):
                pass
            with tag('script', src="https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"):
                pass
        with tag('body'):
            # doc.asis('<div class="container">')
            doc.asis('<div class="card">')
            doc.asis('<div class="card-body">')
            doc.asis('<form class="form-horizontal col-sm-9">')
            for d in data:
                doc.asis(d)
            doc.asis('</form></div></div>')
    
    result = indent(doc.getvalue())
    # print(result)
    with open('test.html', 'w') as HTMLCode:
        HTMLCode.write(result)


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
elem_list = data["regions"]
elem_list = sorted(elem_list, key=lambda k: k["y"])

for element in elem_list:
    doc, tag, text = Doc().tagtext()
    display_attribute = "form-check-inline"
    
    if element["type"] == "Label" or element["type"] == "Heading":
        doc.asis('<div class="col-sm-offset-2 col-sm-10">')
        text(element["content"])
        doc.asis('</div>')
        content.append(doc.getvalue())
    elif element["type"] == "TextBox":
        doc.asis('<div class="form-group">')
        doc.asis('<div class="col-18">')
        doc.stag('input', type = 'text', klass = 'form-control')
        doc.asis('</div></div>')
        content.append(doc.getvalue())
    elif element["type"] == "Password":
        doc.asis('<div class="form-group">')
        doc.asis('<div class="col-18">')
        doc.stag('input', type = 'password', klass = 'form-control')
        doc.asis('</div></div>')
        content.append(doc.getvalue())    
    elif element["type"] == "CheckBox":
        doc.asis('<div class="col-sm-offset-2 col-sm-10">')
        doc.asis('<div class="checkbox">')
        with tag('label'):
            with tag('input', type = 'checkbox' ,value = element['content']):
                text(" " + element["content"])
        doc.asis('</div></div>')
        content.append(doc.getvalue())
    elif element["type"] == "Button":
        doc.asis('<div class="form-group">')        
        doc.asis('<div class="col-sm-offset-2  col-sm-10">')
        with tag('button', klass = 'btn btn-' +random.choice(['primary', 'danger', 'success', 'info'])):
            text(element['content'])
        # doc.stag('button', klass = 'btn btn-default' ,value = element['content'], id = element['id'])
        doc.asis('</div></div>')
        content.append(doc.getvalue())
    elif element["type"] == "RadioButton":
        print("prev_y " + str(prev_y))
        print("prev_h " + str(prev_h))
        print("y " + str(element["y"]))
        print("h " + str(element["height"]))
        
        if ((prev_y + prev_h) < (element["y"]) and consecutive_radio != 0):
            display_attribute = "form-check"
            consecutive_radio = 0
        
        consecutive_radio += 1
        
        doc.asis('<div class=' + display_attribute + '>') 
        doc.asis('<div class="col-sm-offset-2 col-sm-12">')
        doc.asis('<label class="form-check-label" for="radio1">')
        doc.asis('<input type="radio" class="form-check-input" id="radio1" name="optradio" value="' + element["content"] + '">' + element["content"])
        doc.asis('</label>')
        doc.asis('</div></div>')
        content.append(doc.getvalue())
    elif element["type"] == "Image":
        doc.stag('img', src='logo.png', height=element['height'], width= element['width'])
        content.append(doc.getvalue())
    
    prev_y = element["y"]
    prev_h = element["height"]

generate_html(content,'\n'.join(css_list))

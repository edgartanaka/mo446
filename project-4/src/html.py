""""
University of Campinas

MO446 - Computer Vision (Prof. Adin)
Project 4 - Interfaces from sketches

Authors:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
HTML composer based on the patterns detected in the sketch image.
"""

from yattag import Doc
from yattag import indent

class HtmlBuilder(object):
    doc = None
    tag = None
    text = None
    element_idx = 0
    css_text = '''
    body {
      margin:0px;
      background-image:none;
      position:static;
      left:auto;
      width:1000px;
      margin-left:0;
      margin-right:0;
      text-align:left;
    }
    #base {
      position:absolute;
      z-index:0;
    }
    h1,h2,h3,h4,h5 {
      margin: 0;
    }
    '''

    def __init__(self):
        self.doc, self.tag, self.text = Doc().tagtext()
        self.element_idx = 0


    def add_css(self, css):
        self.css_text += css


    def add_image(self, top, left, width, height):
        """
        Creates and adds an image element
        :param top:
        :param left:
        :param width:
        :param height:
        :return:
        """
        id = 'img_' + str(self.element_idx)
        with self.tag('div', id=id):
            self.text('')

        css = '''
            #{} {{
                top: {}px;
                left: {}px;
                width: {}px;
                height: {}px;
                border-width:2px;
                position:absolute;
                background-color: green;
            }}
        '''.format(id, top, left, width, height)
        self.add_css(css)

        self.element_idx += 1


    def add_header(self, top, left, width, height):
        """
        Creates and adds a header element
        :param top:
        :param left:
        :param width:
        :param height:
        :return:
        """
        id = 'header_' + str(self.element_idx)
        with self.tag('div', id=id):
            with self.tag('h3'):
                self.text(
                    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean euismod bibendum laoreet. Proin gravida dolor sit amet lacus accumsan et viverra justo commodo. Proin sodales pulvinar sic tempor. Sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nam fermentum, nulla luctus pharetra vulputate, felis tellus mollis orci, sed rhoncus pronin sapien nunc accuan eget. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean euismod bibendum laoreet. Proin gravida dolor sit amet lacus accumsan et viverra justo commodo. Proin sodales pulvinar sic tempor. Sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nam fermentum, nulla luctus pharetra vulputate, felis tellus mollis orci, sed rhoncus pronin sapien nunc accuan eget.Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean euismod bibendum laoreet. Proin gravida dolor sit amet lacus accumsan et viverra justo commodo. Proin sodales pulvinar sic tempor. Sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nam fermentum, nulla luctus pharetra vulputate, felis tellus mollis orci, sed rhoncus pronin sapien nunc accuan eget.')

        # minimum height is 20px
        height = max(height, 20)

        css = '''
            #{} {{
                top: {}px;
                left: {}px;
                width: {}px;
                height: {}px;
                border-width:2px;
                position:absolute;
                overflow: hidden;
            }}
        '''.format(id, top, left, width, height)
        self.add_css(css)

        self.element_idx += 1


    def add_text(self, top, left, width, height):
        """
        Creates and adds a text element
        :param top:
        :param left:
        :param width:
        :param height:
        :return:
        """
        id = 'text' + str(self.element_idx)
        with self.tag('div', id=id):
            with self.tag('span'):
                self.text(
                    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean euismod bibendum laoreet. Proin gravida dolor sit amet lacus accumsan et viverra justo commodo. Proin sodales pulvinar sic tempor. Sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nam fermentum, nulla luctus pharetra vulputate, felis tellus mollis orci, sed rhoncus pronin sapien nunc accuan eget. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean euismod bibendum laoreet. Proin gravida dolor sit amet lacus accumsan et viverra justo commodo. Proin sodales pulvinar sic tempor. Sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nam fermentum, nulla luctus pharetra vulputate, felis tellus mollis orci, sed rhoncus pronin sapien nunc accuan eget.Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean euismod bibendum laoreet. Proin gravida dolor sit amet lacus accumsan et viverra justo commodo. Proin sodales pulvinar sic tempor. Sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nam fermentum, nulla luctus pharetra vulputate, felis tellus mollis orci, sed rhoncus pronin sapien nunc accuan eget.')

        # minimum height is 20px
        height = max(height, 20)

        css = '''
            #{} {{
                top: {}px;
                left: {}px;
                width: {}px;
                height: {}px;
                border-width:2px;
                position:absolute;
                overflow: hidden;
            }}
        '''.format(id, top, left, width, height)
        self.add_css(css)

        self.element_idx += 1

    def get_html(self, layout):
        self.doc.asis('<!DOCTYPE html>')

        with self.tag('html'):
            with self.tag('body'):
                with self.tag('div', id='base'):
                    for e in layout.elements:
                        if e.type == 'text':
                            self.add_text(e.y, e.x, e.width, e.height)
                        elif e.type == 'header':
                            self.add_header(e.y, e.x, e.width, e.height)
                        elif e.type == 'image':
                            self.add_image(e.y, e.x, e.width, e.height)

            with self.tag('style'):
                self.text(self.css_text)

        return indent(self.doc.getvalue())

    def get_blank_html(self):
        self.doc.asis('<!DOCTYPE html>')

        with self.tag('html'):
            with self.tag('body'):
                self.text('No Element Detected')

        return indent(self.doc.getvalue())


def main():
    html_builder = HtmlBuilder()
    html = html_builder.test()
    f = open("demofile.html", "w")
    f.write(html)
    f.close()

if __name__ == "__main__":
    main()
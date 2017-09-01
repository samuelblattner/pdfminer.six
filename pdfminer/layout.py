from itertools import chain
from typing import List

import math
import re

from .utils import INF
from .utils import Plane
from .utils import get_bound
from .utils import uniq
from .utils import csort
from .utils import fsplit
from .utils import bbox2str
from .utils import matrix2str
from .utils import apply_matrix_pt

import six # Python 2+3 compatibility

##  IndexAssigner
##
class IndexAssigner(object):

    def __init__(self, index=0):
        self.index = index
        return

    def run(self, obj):
        if isinstance(obj, LTTextBox):
            obj.index = self.index
            self.index += 1
        elif isinstance(obj, LTTextGroup):
            for x in obj:
                self.run(x)
        return


##  LAParams
##
class LAParams(object):

    def __init__(self,
                 line_overlap=0.5,
                 char_margin=2.0,
                 line_margin=0.5,
                 word_margin=0.1,
                 boxes_flow=0.5,
                 detect_vertical=False,
                 all_texts=False,

                 # Footnote recognition
                 auto_footnotes=False,
                 footnote_font_size=6.51,
                 footnote_max_def_size=8,
                 footnote_min_def_distance=4,

                 # List recognition
                 auto_lists=False,
                 list_min_def_distance=10,
                 ):
        self.line_overlap = line_overlap
        self.char_margin = char_margin
        self.line_margin = line_margin
        self.word_margin = word_margin
        self.boxes_flow = boxes_flow
        self.detect_vertical = detect_vertical
        self.all_texts = all_texts
        self.auto_footnotes = auto_footnotes
        self.footnote_font_size = footnote_font_size
        self.footnote_max_def_size = footnote_max_def_size
        self.footnote_min_def_distance = footnote_min_def_distance
        self.auto_lists = auto_lists
        self.list_min_def_distance = list_min_def_distance
        return

    def __repr__(self):
        return ('<LAParams: char_margin=%.1f, line_margin=%.1f, word_margin=%.1f all_texts=%r>' %
                (self.char_margin, self.line_margin, self.word_margin, self.all_texts))


##  LTItem
##
class LTItem(object):

    def analyze(self, laparams):
        """Perform the layout analysis."""
        return


##  LTText
##
class LTText(object):

    def __repr__(self):
        return ('<%s %r>' %
                (self.__class__.__name__, self.get_text()))

    def get_text(self):
        raise NotImplementedError


##  LTComponent
##
class LTComponent(LTItem):

    def __init__(self, bbox):
        LTItem.__init__(self)
        self.set_bbox(bbox)
        return

    def __repr__(self):
        return ('<%s %s>' %
                (self.__class__.__name__, bbox2str(self.bbox)))

    # Disable comparison.
    def __lt__(self, _):
        raise ValueError
    def __le__(self, _):
        raise ValueError
    def __gt__(self, _):
        raise ValueError
    def __ge__(self, _):
        raise ValueError

    def set_bbox(self, bbox):
        (x0, y0, x1, y1) = bbox
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = x1-x0
        self.height = y1-y0
        self.bbox = bbox
        return

    def is_empty(self):
        return self.width <= 0 or self.height <= 0

    def is_hoverlap(self, obj):
        assert isinstance(obj, LTComponent), str(type(obj))
        return obj.x0 <= self.x1 and self.x0 <= obj.x1

    def hdistance(self, obj):
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_hoverlap(obj):
            return 0
        else:
            return min(abs(self.x0-obj.x1), abs(self.x1-obj.x0))

    def hoverlap(self, obj):
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_hoverlap(obj):
            return min(abs(self.x0-obj.x1), abs(self.x1-obj.x0))
        else:
            return 0

    def is_voverlap(self, obj):
        assert isinstance(obj, LTComponent), str(type(obj))
        return obj.y0 <= self.y1 and self.y0 <= obj.y1

    def vdistance(self, obj):
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_voverlap(obj):
            return 0
        else:
            return min(abs(self.y0-obj.y1), abs(self.y1-obj.y0))

    def voverlap(self, obj):
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_voverlap(obj):
            return min(abs(self.y0-obj.y1), abs(self.y1-obj.y0))
        else:
            return 0


##  LTCurve
##
class LTCurve(LTComponent):

    def __init__(self, linewidth, pts, stroke = False, fill = False, evenodd = False, stroking_color = None, non_stroking_color = None):
        LTComponent.__init__(self, get_bound(pts))
        self.pts = pts
        self.linewidth = linewidth
        self.stroke = stroke
        self.fill = fill
        self.evenodd = evenodd
        self.stroking_color = stroking_color
        self.non_stroking_color = non_stroking_color
        return

    def get_pts(self):
        return ','.join('%.3f,%.3f' % p for p in self.pts)


##  LTLine
##
class LTLine(LTCurve):

    def __init__(self, linewidth, p0, p1, stroke = False, fill = False, evenodd = False, stroking_color = None, non_stroking_color = None):
        LTCurve.__init__(self, linewidth, [p0, p1], stroke, fill, evenodd, stroking_color, non_stroking_color)
        self.is_horizontal = p0[1] == p1[1]
        return


##  LTRect
##
class LTRect(LTCurve):

    def __init__(self, linewidth, bbox, stroke = False, fill = False, evenodd = False, stroking_color = None, non_stroking_color = None):
        (x0, y0, x1, y1) = bbox
        LTCurve.__init__(self, linewidth, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], stroke, fill, evenodd, stroking_color, non_stroking_color)
        return


##  LTImage
##
class LTImage(LTComponent):

    def __init__(self, name, stream, bbox):
        LTComponent.__init__(self, bbox)
        self.name = name
        self.stream = stream
        self.srcsize = (stream.get_any(('W', 'Width')),
                        stream.get_any(('H', 'Height')))
        self.imagemask = stream.get_any(('IM', 'ImageMask'))
        self.bits = stream.get_any(('BPC', 'BitsPerComponent'), 1)
        self.colorspace = stream.get_any(('CS', 'ColorSpace'))
        if not isinstance(self.colorspace, list):
            self.colorspace = [self.colorspace]
        return

    def __repr__(self):
        return ('<%s(%s) %s %r>' %
                (self.__class__.__name__, self.name,
                 bbox2str(self.bbox), self.srcsize))


##  LTAnno
##
class LTAnno(LTItem, LTText):

    def __init__(self, text):
        self._text = text
        return

    def get_text(self):
        return self._text


##  LTChar
##
class LTChar(LTComponent, LTText):

    def __init__(self, matrix, font, fontsize, scaling, rise,
                 text, textwidth, textdisp):
        LTText.__init__(self)
        self._text = text
        self.matrix = matrix
        self.fontname = font.fontname
        self.font = font
        self.fontsize = fontsize
        self.scaling = scaling
        self.rise = rise
        self.textwidth = textwidth
        self.textdisp = textdisp
        self.adv = textwidth * fontsize * scaling
        # compute the boundary rectangle.
        if font.is_vertical():
            # vertical
            width = font.get_width() * fontsize
            (vx, vy) = textdisp
            if vx is None:
                vx = width * 0.5
            else:
                vx = vx * fontsize * .001
            vy = (1000 - vy) * fontsize * .001
            tx = -vx
            ty = vy + rise
            bll = (tx, ty+self.adv)
            bur = (tx+width, ty)
        else:
            # horizontal
            height = font.get_height() * fontsize
            descent = font.get_descent() * fontsize
            ty = descent + rise
            bll = (0, ty)
            bur = (self.adv, ty+height)
        (a, b, c, d, e, f) = self.matrix
        self.upright = (0 < a*d*scaling and b*c <= 0)
        (x0, y0) = apply_matrix_pt(self.matrix, bll)
        (x1, y1) = apply_matrix_pt(self.matrix, bur)
        if x1 < x0:
            (x0, x1) = (x1, x0)
        if y1 < y0:
            (y0, y1) = (y1, y0)
        LTComponent.__init__(self, (x0, y0, x1, y1))
        if font.is_vertical():
            self.size = self.width
        else:
            self.size = self.height
        return

    def __repr__(self):
        return ('<%s %s matrix=%s font=%r adv=%s text=%r>' %
                (self.__class__.__name__, bbox2str(self.bbox),
                 matrix2str(self.matrix), self.fontname, self.adv,
                 self.get_text()))

    def get_text(self):
        return self._text

    def is_compatible(self, obj):
        """Returns True if two characters can coexist in the same line."""
        return True


class LTFootnoteMarker(LTComponent, LTText):

    def __init__(self, text, bbox, *args, **kwargs):
        super(LTFootnoteMarker, self).__init__(bbox)
        self.text = text

    def get_text(self):
        return self.text


##  LTContainer
##
class LTContainer(LTComponent):

    def __init__(self, bbox):
        LTComponent.__init__(self, bbox)
        self._objs = []
        return

    def __iter__(self):
        return iter(self._objs)

    def __len__(self):
        return len(self._objs)

    def add(self, obj):
        self._objs.append(obj)
        return

    def remove(self, obj):
        if obj in self._objs:
            self._objs.remove(obj)

    def extend(self, objs):
        for obj in objs:
            self.add(obj)
        return

    def analyze(self, laparams):
        for obj in self._objs:
            obj.analyze(laparams)
        return


##  LTExpandableContainer
##
class LTExpandableContainer(LTContainer):

    def __init__(self):
        LTContainer.__init__(self, (+INF, +INF, -INF, -INF))
        return

    def add(self, obj):
        LTContainer.add(self, obj)
        self.set_bbox((min(self.x0, obj.x0), min(self.y0, obj.y0),
                       max(self.x1, obj.x1), max(self.y1, obj.y1)))
        return

    def remove(self, obj):
        LTContainer.remove(self, obj)
        all_x = []
        all_y = []
        for o in self._objs:
            all_x.extend([o.x0, o.x1])
            all_y.extend([o.y0, o.y1])

        if len(self._objs) > 0:
            self.set_bbox((min(all_x), min(all_y), max(all_x), max(all_y)))
        else:
            self.set_bbox((0, 0, 0, 0))


##  LTTextContainer
##
class LTTextContainer(LTExpandableContainer, LTText):

    def __init__(self):
        LTText.__init__(self)
        LTExpandableContainer.__init__(self)
        return

    def get_text(self):
        return ''.join(obj.get_text() for obj in self if isinstance(obj, LTText))


##  LTTextLine
##
class LTTextLine(LTTextContainer):

    min_super_script_baseline_distance_percent = 10
    cur_footnote_number = 0

    def __init__(self, word_margin):
        LTTextContainer.__init__(self)
        self.word_margin = word_margin
        return

    def __repr__(self):
        return ('<%s %s %r>' %
                (self.__class__.__name__, bbox2str(self.bbox),
                 self.get_text()))

    def __get_superscript(self, letters: List[LTChar]) -> str:
        """
        Checks if a given list of LTChar all have a min y position that would define
        them as 'superscripted'. Returns True if all of the Characters fulfill this criterion.

        :param line: PDF line that contains the characters and that is a reference for the baseline y value.
        :param letters: List of Characters that should be checked
        :return: True if all Characters are superscripted, False if not
        """

        base_line = self.y0
        line_height = self.y1 - base_line
        min_super_script_y_pos = base_line + line_height * self.min_super_script_baseline_distance_percent / 100

        output = ''
        for letter in letters:

            if letter.y0 < min_super_script_y_pos:
                output += '¬' if letter.get_text() != ' ' else ' '
            else:
                output += letter.get_text()
        return output

    def analyze(self, laparams):

        LTTextContainer.analyze(self, laparams)

        if laparams.auto_footnotes:
            superscripted = self.__get_superscript([obj for obj in self._objs if isinstance(obj, LTChar)])

            matches = re.finditer(r'(?<!^)(?<!\d)(\d+)', superscripted)
            for match in matches:
                letters = [letter for letter in self._objs[match.start():match.end()] if isinstance(letter, LTChar)]
                try:
                    footnote_number = int(match.group(1))
                except ValueError:
                    continue

                if LTTextLine.cur_footnote_number == -1:
                    LTTextLine.cur_footnote_number = footnote_number
                if abs(footnote_number - LTTextLine.cur_footnote_number) > 20:
                    continue
                LTTextLine.cur_footnote_number = footnote_number

                if len(letters) > 0:
                    self._objs = self._objs[:match.start()] + [
                        LTFootnoteMarker(
                            match.group(1),
                            (
                                letters[0].y0, letters[0].x0,
                                letters[-1].y1, letters[-1].x1
                            )
                        )] + self._objs[match.end():]

        LTContainer.add(self, LTAnno('\n'))
        return

    def find_neighbors(self, plane, ratio, **kwargs):
        raise NotImplementedError


class LTTextLineHorizontal(LTTextLine):

    def __init__(self, word_margin):
        LTTextLine.__init__(self, word_margin)
        self._x1 = +INF
        return

    def add(self, obj):
        if isinstance(obj, LTChar) and self.word_margin:
            margin = self.word_margin * max(obj.width, obj.height)
            if self._x1 < obj.x0-margin:
                LTContainer.add(self, LTAnno(' '))
        self._x1 = obj.x1
        LTTextLine.add(self, obj)
        return

    def is_footnote(self, laparams, surrounding_objects):
        """
        Checks if this textline fulfills the criteria to be a footnote (i.e. starts with a small
        number).

        :param laparams Params
        :param surrounding_objects: Objects that surround the current line
        :return: True if this line is a footnote, False otherwise
        """
        first_letter = self._objs[0]
        footnote_font_size = 999
        font_size = max([l.matrix[0] for l in self._objs[:20] if isinstance(l, LTChar)])
        first_letters_is_number = re.search(r'^(\d{1,4})', self.get_text())

        if first_letters_is_number:
            footnote_font_size = max([letter.matrix[0] for letter in self._objs[first_letters_is_number.start(): first_letters_is_number.end()]])

        font_size_in_range = font_size <= laparams.footnote_max_def_size + 0.3 and abs(footnote_font_size - laparams.footnote_font_size) < 0.3

        too_close_objs = [
            obj for obj in surrounding_objects
            if obj.x0 - first_letter.x1 < laparams.footnote_min_def_distance and abs(obj.y1 - first_letter.y1) < 1 and obj is not self
        ]

        return font_size_in_range and first_letters_is_number and len(too_close_objs) == 0

    def is_list_element(self, laparams, surrounding_objects):
        """
        Checks if this textline fulfills the criteria to be a footnote (i.e. starts with a small
        number).

        :param laparams Params
        :param surrounding_objects: Objects that surround the current line
        :return: True if this line is a footnote, False otherwise
        """
        first_letter = self._objs[0]
        first_letters = self.get_text()[:10]
        first_letter_font_size = first_letter.matrix[0]

        first_letter_is_list_index = re.search(r'^([a-z\d]{1,3}\.)+', first_letters) is not None

        has_objs_with_same_x = len([obj for obj in surrounding_objects if obj.x0 == self.x0]) > 0
        has_objs_with_same_y = len([obj for obj in surrounding_objects if abs(obj.y1-self.y1) < 0.5 and obj.x1 > self.x1]) > 0
        has_too_close_objs = len([
            obj for obj in surrounding_objects
            if obj.x0 - first_letter.x1 < laparams.list_min_def_distance and abs(obj.y1 - first_letter.y1) < 1 and obj is not self
        ]) > 0

        return first_letter_is_list_index and has_objs_with_same_x and has_objs_with_same_y and not has_too_close_objs and first_letter_font_size > 8

    def find_neighbors(self, plane, laparams, **kwargs):

        auto_footnotes = laparams.auto_footnotes
        auto_lists = laparams.auto_lists

        # 1. Check if this line is a footnote (if required)
        dv = laparams.line_margin*self.height

        horizontal_scan_range = laparams.footnote_min_def_distance + 20 if auto_footnotes else 0
        objs = list(plane.find((self.x0, self.y0-dv*20, self.x1 + horizontal_scan_range, self.y1+dv)))
        is_footnote = self.is_footnote(laparams, objs) if auto_footnotes else False
        is_list_element = False
        dh = horizontal_scan_range + 10 if is_footnote else dv

        if not is_footnote:
            horizontal_scan_range = laparams.list_min_def_distance + 4 if auto_lists else 0
            objs = list(plane.find((self.x0, self.y0 - dv, self.x1 + horizontal_scan_range, self.y1 + dv)))
            is_list_element = self.is_list_element(laparams, objs) if auto_lists else False
            dh = horizontal_scan_range + 10 if auto_lists else dv

        # 2. Scan for surrounding footnotes. We must not collect any
        # lines that belong to other footnote definitions
        dist_lines_above = []
        dist_lines_below = []

        if is_footnote:
            for obj in objs:

                if obj.is_footnote(laparams, objs) and obj is not self and abs(obj.x0 - self.x0) < 1:

                    dist = obj.y1 - self.y1
                    if dist >= 0:
                        dist_lines_above.append(dist)
                    else:
                        dist_lines_below.append(dist)

        elif is_list_element:
            for obj in objs:
                if obj.is_list_element(laparams, objs) and obj is not self:
                    dist = obj.y1 - self.y1
                    if dist >= 0:
                        dist_lines_above.append(dist)
                    else:
                        dist_lines_below.append(dist)

        max_dy_above = min(dist_lines_above) if len(dist_lines_above) > 0 else 40
        max_dy_below = max(dist_lines_below) if len(dist_lines_below) > 0 else -40

        return [obj for obj in objs
                if (isinstance(obj, LTTextLineHorizontal) and
                    abs(obj.height-self.height) < dv and
                    (abs(obj.x0-self.x0) < dh or abs(obj.x1-self.x1) < dh)) and
                    obj.x0 >= self.x0 and
                    obj.y1 <= self.y1 and
                    max_dy_below < obj.y1 - self.y1 < max_dy_above or
                    obj is self
               ], is_footnote, is_list_element


class LTTextLineVertical(LTTextLine):

    def __init__(self, word_margin):
        LTTextLine.__init__(self, word_margin)
        self._y0 = -INF
        return

    def add(self, obj):
        if isinstance(obj, LTChar) and self.word_margin:
            margin = self.word_margin * max(obj.width, obj.height)
            if obj.y1+margin < self._y0:
                LTContainer.add(self, LTAnno(' '))
        self._y0 = obj.y0
        LTTextLine.add(self, obj)
        return

    def find_neighbors(self, plane, ratio, footnote_font_size=6.5):
        d = ratio*self.width
        objs = plane.find((self.x0-d, self.y0, self.x1+d, self.y1))
        return [obj for obj in objs
                if (isinstance(obj, LTTextLineVertical) and
                    abs(obj.width-self.width) < d and
                    (abs(obj.y0-self.y0) < d or
                     abs(obj.y1-self.y1) < d))]


##  LTTextBox
##
##  A set of text objects that are grouped within
##  a certain rectangular area.
##
class LTTextBox(LTTextContainer):

    def __init__(self):
        LTTextContainer.__init__(self)
        self.index = -1
        return

    def __repr__(self):
        return ('<%s(%s) %s %r>' %
                (self.__class__.__name__,
                 self.index, bbox2str(self.bbox), self.get_text()))


class LTTextBoxHorizontal(LTTextBox):

    def analyze(self, laparams):
        LTTextBox.analyze(self, laparams)
        self._objs = csort(self._objs, key=lambda obj: -obj.y1)
        return

    def get_writing_mode(self):
        return 'lr-tb'


class LTFootnote(LTTextBoxHorizontal):

    def analyze(self, laparams):
        LTTextBox.analyze(self, laparams)
        self._objs = csort(self._objs, key=lambda obj: -obj.y1)
        return

    def get_writing_mode(self):
        return 'lr-tb'


class LTTextBoxVertical(LTTextBox):

    def analyze(self, laparams):
        LTTextBox.analyze(self, laparams)
        self._objs = csort(self._objs, key=lambda obj: -obj.x1)
        return

    def get_writing_mode(self):
        return 'tb-rl'


##  LTTextGroup
##
class LTTextGroup(LTTextContainer):

    def __init__(self, objs):
        LTTextContainer.__init__(self)
        self.extend(objs)
        return


class LTTextGroupLRTB(LTTextGroup):

    def analyze(self, laparams):
        LTTextGroup.analyze(self, laparams)
        # reorder the objects from top-left to bottom-right.
        self._objs = csort(self._objs, key=lambda obj:
                           (1-laparams.boxes_flow)*(obj.x0) -
                           (1+laparams.boxes_flow)*(obj.y0+obj.y1))
        return


class LTTextGroupTBRL(LTTextGroup):

    def analyze(self, laparams):
        LTTextGroup.analyze(self, laparams)
        # reorder the objects from top-right to bottom-left.
        self._objs = csort(self._objs, key=lambda obj:
                           -(1+laparams.boxes_flow)*(obj.x0+obj.x1)
                           - (1-laparams.boxes_flow)*(obj.y1))
        return


class LTTableCell:

    textlines = []

    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.textlines = []

    def add_textline(self, text_line):
        if text_line not in self.textlines:
            self.textlines.append(text_line)

    def get_text(self):
        return '\n'.join([tl.get_text() for tl in self.textlines])


class LTTableRow():
    cells = []

    def __init__(self):
        self.cells = []

    def add_cell(self, cell):
        if cell not in self.cells:
            self.cells.append(cell)


class LTTable():

    rows = []

    def __init__(self, x, y, width, height):
        self.rows = []
        self.x = x
        self.x0 = x
        self.x1 = x
        self.y = y
        self.y0 = y
        self.y1 = y
        self.width = width
        self.height = height

    def add_row(self, row):
        if row not in self.rows:
            self.rows.append(row)

class LTIntersection:

    x = 0
    y = 0
    weight = 0
    neighbors = None
    predecessor = None
    can_close_cycle = False

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.weight = 0
        self.predecessor = None
        self.can_close_cycle = False
        self.neighbors = []

    def __str__(self):
        return '{}/{}'.format(self.x, self.y)

    @staticmethod
    def find_shortest_cycle(target_intersection, start_intersection, remaining_intersections=None):

        for neighbor in start_intersection.neighbors:

            if neighbor.can_close_cycle:
                return

            if start_intersection == target_intersection:
                neighbor.can_close_cycle = True

            if neighbor in remaining_intersections:

                neighbor_weight = abs(neighbor.x - start_intersection.x) if neighbor.y == start_intersection.y else \
                    abs(neighbor.y - start_intersection.y) + start_intersection.weight

                if neighbor_weight <= neighbor.weight:
                    neighbor.weight = neighbor_weight
                    neighbor.predecessor = start_intersection

    @staticmethod
    def __get_neighbor_for_direction(origin, neighbors, direction):
        for neighbor in neighbors:

            if direction == 0 and abs(neighbor.y - origin.y) < 0.5 and neighbor.x > origin.x:
                return neighbor
            elif direction == 1 and abs(neighbor.x - origin.x) < 0.5 and neighbor.y < origin.y:
                return neighbor
            elif direction == 2 and abs(neighbor.y - origin.y) < 0.5 and neighbor.x < origin.x:
                return neighbor
            elif direction == 3 and abs(neighbor.x - origin.x) < 0.5 and neighbor.y > origin.y:
                return neighbor

        return None

    @staticmethod
    def travel_intersections(start_intersection, direction):

        next_node = start_intersection
        last_node = next_node
        circle = [next_node]
        last_direction = 0
        direction_changes = 0

        while next_node:
            next_node = LTIntersection.__get_neighbor_for_direction(next_node, next_node.neighbors, direction)

            if not next_node and direction == 0:
                return None

            if next_node and direction >= 3 and next_node == start_intersection:
                return circle

            if not next_node and direction <= 4:
                next_node = last_node
                direction -= 1
                continue

            if direction != last_direction:
                circle.append(last_node)


            last_direction = direction
            direction_changes += 1

            # Try to change to the next direction to get an as small circle as possible
            direction += 1
            last_node = next_node

        return next_node


class LTTabStop:

    def __init__(self, x_pos, align='left', top_most=560, bottom_most=0, left_most=0, right_most=100, num_elements=None):
        self.align = align
        self.x = x_pos
        self.y = x_pos
        self.top_most = top_most
        self.bottom_most = bottom_most
        self.left_most = left_most
        self.right_most = right_most
        self.num_elements = num_elements


##  LTLayoutContainer
##
class LTLayoutContainer(LTContainer):

    def __init__(self, bbox):
        LTContainer.__init__(self, bbox)
        self.groups = None
        self.tabstops = []
        return

    # group_objects: group text object to textlines.
    def group_objects(self, laparams, objs):
        obj0 = None
        line = None
        for obj1 in objs:
            if obj0 is not None:
                # halign: obj0 and obj1 is horizontally aligned.
                #
                #   +------+ - - -
                #   | obj0 | - - +------+   -
                #   |      |     | obj1 |   | (line_overlap)
                #   +------+ - - |      |   -
                #          - - - +------+
                #
                #          |<--->|
                #        (char_margin)
                halign = (obj0.is_compatible(obj1) and
                          obj0.is_voverlap(obj1) and
                          (min(obj0.height, obj1.height) * laparams.line_overlap <
                           obj0.voverlap(obj1)) and
                          (obj0.hdistance(obj1) <
                           max(obj0.width, obj1.width) * laparams.char_margin))

                # valign: obj0 and obj1 is vertically aligned.
                #
                #   +------+
                #   | obj0 |
                #   |      |
                #   +------+ - - -
                #     |    |     | (char_margin)
                #     +------+ - -
                #     | obj1 |
                #     |      |
                #     +------+
                #
                #     |<-->|
                #   (line_overlap)
                valign = (laparams.detect_vertical and
                          obj0.is_compatible(obj1) and
                          obj0.is_hoverlap(obj1) and
                          (min(obj0.width, obj1.width) * laparams.line_overlap <
                           obj0.hoverlap(obj1)) and
                          (obj0.vdistance(obj1) <
                           max(obj0.height, obj1.height) * laparams.char_margin))

                if ((halign and isinstance(line, LTTextLineHorizontal)) or
                    (valign and isinstance(line, LTTextLineVertical))):
                    line.add(obj1)
                elif line is not None:
                    yield line
                    line = None
                else:
                    if valign and not halign:
                        line = LTTextLineVertical(laparams.word_margin)
                        line.add(obj0)
                        line.add(obj1)
                    elif halign and not valign:
                        line = LTTextLineHorizontal(laparams.word_margin)
                        line.add(obj0)
                        line.add(obj1)
                    else:
                        line = LTTextLineHorizontal(laparams.word_margin)
                        line.add(obj0)
                        yield line
                        line = None
            obj0 = obj1
        if line is None:
            line = LTTextLineHorizontal(laparams.word_margin)
            line.add(obj0)
        yield line
        return

    def __any_points_match(self, points1, points2, tolerance=1):
        for point1 in points1:
            for point2 in points2:
                if abs(point1[0] - point2[0]) <= tolerance and abs(point1[1] - point2[1]) <= tolerance:
                    return True

    def __lines_point_in_same_direction(self, lines):
        horizontal = None

        for line in lines:
            if horizontal is None:
                horizontal = line.is_horizontal
            elif horizontal != line.is_horizontal:
                return False

        return True

    def __group_prolonged_lines(self, lines):
        # Glue prolonged lines together
        line_groups = {}
        added_lines = []
        for line in lines:
            for other_line in lines:
                if line is not other_line:
                    if not self.__any_points_match([line.pts[0], line.pts[1]], [other_line.pts[0], other_line.pts[1]]):
                        continue

                    if self.__lines_point_in_same_direction([line, other_line]):
                        if other_line not in added_lines:
                            added_lines.append(other_line)
                        if line not in added_lines:
                            added_lines.append(line)

                        line_groups.setdefault(line, []).append(other_line)

                        if other_line in line_groups:
                            line_groups[line].extend(line_groups.pop(other_line))

        for line in added_lines:
            lines.remove(line)

        grouped = []
        for key_line, grouped_lines in line_groups.items():
            grouped_lines.append(key_line)
            grouped.extend(grouped_lines)
            all_x_coords = list(chain(*[(line.pts[0][0], line.pts[1][0]) for line in grouped_lines]))
            all_y_coords = list(chain(*[(line.pts[0][1], line.pts[1][1]) for line in grouped_lines]))
            fused_line = LTLine(
                1,
                (min(all_x_coords), min(all_y_coords)),
                (max(all_x_coords), max(all_y_coords))
            )

            lines.append(fused_line)

        return lines

    def __calculate_line_intersection(self, line1, line2):
        if line1.is_horizontal == line2.is_horizontal:
                return None, None

        # Horizontal line 1

        if line1.is_horizontal:
            line1_max_x = max(line1.pts[0][0] + 1, line1.pts[1][0] + 1)
            line1_min_x = min(line1.pts[0][0] - 1, line1.pts[1][0] - 1)
            line1_y = line1.pts[0][1]
            line2_max_y = max(line2.pts[0][1] + 1, line2.pts[1][1] + 1)
            line2_min_y = min(line2.pts[0][1] - 1, line2.pts[1][1] - 1)
            line2_x = line2.pts[0][0]

            dx = line1_max_x - line2_x
            dx2 = line2_x - line1_min_x
            dy = line2_max_y - line1_y
            dy2 = line1_y - line2_min_y

            if dx >= 0 and dx2 >= 0 and dy >= 0 and dy2 >= 0:
                return line2_x, line1_y

        # Vertical line 1
        else:
            line1_x = line1.pts[0][0]
            line1_max_y = max(line1.pts[0][1] + 1, line1.pts[1][1] + 1)
            line1_min_y = min(line1.pts[0][1] - 1, line1.pts[1][1] - 1)
            line2_y = line2.pts[0][1]
            line2_max_x = max(line2.pts[0][0] + 1, line2.pts[1][0] + 1)
            line2_min_x = min(line2.pts[0][0] - 1, line2.pts[1][0] - 1)

            dx = line2_max_x - line1_x
            dx2 = line1_x - line2_min_x
            dy = line1_max_y - line2_y
            dy2 = line2_y - line1_min_y

            if dx >= 0 and dx2 >= 0 and dy >= 0 and dy2 >= 0:
                return line1_x, line2_y

        return None, None

    def create_tables(self, laparams, textlines, rects):
        """
        Tries to construct layout tables using LTRect objects and Textline objects.

        :param laparams: LAParams
        :param textlines: Textlines in the page
        :param rects: LTRects in the page
        :return: LTTable object
        """

        # Convert to lines. Any rectangle that is either smaller than 1pt in width or height
        # classifies as line.
        lines = []
        has_vertical_lines = False
        for rect in rects:

            # Horizontal
            if rect.height < 1 and rect.width > rect.height:
                lines.append(LTLine(1, (rect.x0, rect.y1), (rect.x1, rect.y1)))

            # Vertical
            if rect.width < 1 and rect.height > rect.width:
                lines.append(LTLine(1, (rect.x0, rect.y0), (rect.x0, rect.y1)))
                has_vertical_lines = True

        if len(lines) <= 3:
            return [], textlines

        lines = self.__group_prolonged_lines(lines)

        # Calculate imaginary lines
        if not has_vertical_lines:

            top_most_implicit = 0
            bottom_most_implicit = 999

            for tabstop in self.tabstops:
                if tabstop.align == 'left':
                    if tabstop.top_most > top_most_implicit:
                        top_most_implicit = tabstop.top_most
                    if tabstop.bottom_most < bottom_most_implicit:
                        bottom_most_implicit = tabstop.bottom_most

            for tabstop in self.tabstops:
                if tabstop.align == 'left':
                    if tabstop.top_most < top_most_implicit:
                        tabstop.top_most = top_most_implicit
                    if tabstop.bottom_most > bottom_most_implicit:
                        tabstop.bottom_most = bottom_most_implicit

                    for textline in textlines:
                        if textline.x0 + 2 < tabstop.x and textline.x1 > tabstop.x or textline._objs[0].matrix[0] >= 10:
                            tabstop_middle = (tabstop.top_most - tabstop.bottom_most) / 2 + tabstop.bottom_most
                            if tabstop.top_most > textline.y0 > tabstop_middle:
                                tabstop.top_most = textline.y0
                            elif tabstop_middle > textline.y1 > tabstop.bottom_most:
                                tabstop.bottom_most = textline.y1

                    lines.append(LTLine(1, (tabstop.x, tabstop.top_most), (tabstop.x, tabstop.bottom_most)))

            max_x = 0
            for line in lines:
                right_x = max(line.pts[0][0], line.pts[1][0])
                if line.is_horizontal and right_x > max_x:
                    max_x = right_x

            lines.append(LTLine(1, (max_x - 1, bottom_most_implicit), (max_x - 1, top_most_implicit)))

        cur_num_elements = 999
        for tabstop in sorted(self.tabstops, key=lambda t: -t.y):
            if cur_num_elements <= tabstop.num_elements and tabstop.align == 'top':
                for line in lines:
                    if line.is_horizontal and abs(line.pts[0][1] - tabstop.y) < 10:
                        break
                else:
                    lines.append(LTLine(1, (tabstop.left_most - 1, tabstop.y), (tabstop.right_most + 1, tabstop.y)))

            cur_num_elements = tabstop.num_elements

        # Calculate every line's intersections

        intersections = []

        for line in lines:

            # If the line is horizontal, only compare vertical lines and sort them by x
            # so that we can drive along the horizontal line and set the intersections
            if line.is_horizontal:
                other_lines = sorted([vert_line for vert_line in lines if not vert_line.is_horizontal], key=lambda l: l.pts[0][0])

            # Otherwise, only use horizontal lines and order them by y.
            else:
                other_lines = sorted([horz_line for horz_line in lines if horz_line.is_horizontal], key=lambda l: l.pts[0][1])

            previous_intersection = None
            for other_line in other_lines:
                if line is not other_line:
                    ix, iy = self.__calculate_line_intersection(line, other_line)
                    if ix and iy:
                        new_intersection = None
                        for intersection in intersections:
                            if abs(intersection.x - ix) < 0.5 and abs(intersection.y - iy) < 0.5:
                                new_intersection = intersection
                                break

                        if not new_intersection:
                            new_intersection = LTIntersection(ix, iy)
                            intersections.append(new_intersection)

                        if previous_intersection:
                            if previous_intersection not in new_intersection.neighbors:
                                new_intersection.neighbors.append(previous_intersection)

                            if new_intersection not in previous_intersection.neighbors:
                                previous_intersection.neighbors.append(new_intersection)

                        previous_intersection = new_intersection

        # Form cells
        cells = []

        for i, intersection in enumerate(intersections):

            circle = LTIntersection.travel_intersections(intersection, 0)

            if circle:
                all_inter_x = [inter.x for inter in circle]
                all_inter_y = [inter.y for inter in circle]
                cells.append(LTTableCell(min(all_inter_x), min(all_inter_y), max(all_inter_x), max(all_inter_y)))

        row_tops = []
        # Group cells to rows
        for cell in cells:
            if cell.y1 not in row_tops:
                row_tops.append(cell.y1)

        # Guides
        x_guides = []
        y_guides = []
        for cell in cells:
            if cell.x0 not in x_guides:
                x_guides.append(cell.x0)
            if cell.x1 not in x_guides:
                x_guides.append(cell.x1)
            if cell.y1 not in y_guides:
                y_guides.append(cell.y1)
            if cell.y0 not in y_guides:
                y_guides.append(cell.y0)

        if len(x_guides) == 0 or len(y_guides) == 0:
            return [], textlines

        width = 0
        height = 0
        if len(x_guides) > 0:
            width = max(x_guides) - min(x_guides)
        if len(y_guides) > 0:
            height = max(y_guides) - min(y_guides)

        # Spans
        x_guides = sorted(x_guides)
        y_guides = sorted(y_guides, reverse=True)
        for cell in cells:
            start_guide_x_index = x_guides.index(cell.x0)
            end_guide_x_index = x_guides.index(cell.x1)
            cell.span_x = end_guide_x_index - start_guide_x_index
            cell.width = '{}%'.format((cell.x1 - cell.x0) / width * 100)

            start_guide_y_index = y_guides.index(cell.y1)
            end_guide_y_index = y_guides.index(cell.y0)
            cell.span_y = end_guide_y_index - start_guide_y_index

        table = LTTable(min(x_guides), max(y_guides), width, height)
        for row_top in sorted(row_tops, reverse=True):
            row = LTTableRow()
            table.add_row(row)
            for cell in sorted(cells, key=lambda c: (c.y1, c.x0)):
                if cell.y1 == row_top:
                    row.add_cell(cell)

        # assign textlines
        remaining_textlines = []
        for textline in textlines:
            for cell in cells:
                if textline.x0 + 1 > cell.x0 and textline.x1 < cell.x1 and textline.y1 - 1 < cell.y1 and (textline.y0 > cell.y0 or abs(textline.y0 - cell.y0) < 3):
                    cell.add_textline(textline)
                    break
            else:
                remaining_textlines.append(textline)

        return [table], remaining_textlines

    def create_tabstops(self, laparams, textlines):

        left_aligned_x = {}
        top_aligned_y = {}
        right_aligned_x = {}
        bottom_aligned_y = {}
        round_digits = 1
        overlap = 2
        for textline in textlines:
            left_aligned_x.setdefault(round(textline.x0, round_digits), []).append(textline)
            right_aligned_x.setdefault(round(textline.x1, round_digits), []).append(textline)
            top_aligned_y.setdefault(round(textline.y1, round_digits), []).append(textline)
            bottom_aligned_y.setdefault(round(textline.y0, round_digits), []).append(textline)

        for tabstop_x, elements in left_aligned_x.items():
            if len(elements) >= 2:
                self.tabstops.append(LTTabStop(tabstop_x, 'left', top_most=max([tl.y1 for tl in elements]) + overlap, bottom_most=min([tl.y0 for tl in elements]) - overlap, num_elements=len(elements)))
                for element in elements:
                    try:
                        del right_aligned_x[round(element.x1, round_digits)]
                    except KeyError:
                        pass

        for tabstop_x, elements in right_aligned_x.items():
            if len(elements) >= 2:
                self.tabstops.append(LTTabStop(round(tabstop_x, round_digits), 'right', top_most=max([tl.y1 for tl in elements]) + overlap, bottom_most=min([tl.y0 for tl in elements]) - overlap, num_elements=len(elements)))

        # find align gaps
        last_y1 = None
        last_y_dist = 0
        gaps = {}
        for left_tab_stop in self.tabstops:
            if left_tab_stop.align == 'left':
                for textline in sorted(textlines, key=lambda tl: tl.y1):
                    if round(textline.x0, round_digits) == left_tab_stop.x:
                        if last_y1 and last_y_dist and last_y1 - textline.y1 > last_y_dist * 2:
                            gaps.setdefault(round(textline.y1, round_digits), []).append(textline)

                        last_y1 = textline.y1
                        last_y_dist = last_y1 - textline.y1

        for tabstop_y, elements in top_aligned_y.items():
            if len(elements) >= 3 and tabstop_y in gaps:
                self.tabstops.append(LTTabStop(tabstop_y, 'top', left_most=min([tl.x0 for tl in elements]) - overlap, right_most=max([tl.x1 for tl in elements]) + overlap, num_elements=len(elements)))
                for element in elements:
                    try:
                        del bottom_aligned_y[round(element.y0, round_digits)]
                    except KeyError:
                        pass

        for tabstop_y, elements in bottom_aligned_y.items():
            if len(elements) >= 3:
                self.tabstops.append(LTTabStop(tabstop_y, 'bottom', left_most=min([tl.x0 for tl in elements]) - overlap, right_most=max([tl.x1 for tl in elements]) + overlap, num_elements=len(elements)))

    # group_textlines: group neighboring lines to textboxes.
    def group_textlines(self, laparams, lines):
        plane = Plane(self.bbox)
        plane.extend(lines)
        boxes = {}
        autogroup_lines = []
        do_auto_footnote = laparams.auto_footnotes
        do_auto_list = laparams.auto_lists
        for line in lines:

            neighbors, is_footnote, is_list = line.find_neighbors(plane, laparams)

            if line not in neighbors or line in autogroup_lines: continue
            members = []
            for obj1 in neighbors:

                # if obj1 in autogroup_lines:
                #     continue

                if obj1 is not line and do_auto_footnote and obj1.is_footnote(laparams, plane):
                    continue

                if obj1 is not line and do_auto_list and obj1.is_list_element(laparams, plane):
                    continue

                members.append(obj1)
                if obj1 in boxes:
                    if is_footnote or is_list:
                        boxes[obj1].remove(obj1)
                        boxes.pop(obj1)
                    else:
                        members.extend(boxes.pop(obj1))

            if isinstance(line, LTTextLineHorizontal):
                if is_list or is_footnote:
                    autogroup_lines.extend(members)
                if is_footnote:
                    box = LTFootnote()
                else:
                    box = LTTextBoxHorizontal()
            else:
                box = LTTextBoxVertical()

            for obj in uniq(members):
                box.add(obj)
                boxes[obj] = box
        done = set()
        for line in lines:
            if line not in boxes: continue
            box = boxes[line]
            if box in done:
                continue
            done.add(box)
            if not box.is_empty():
                yield box
        return

    # group_textboxes: group textboxes hierarchically.
    def group_textboxes(self, laparams, boxes):
        assert boxes, str((laparams, boxes))

        def dist(obj1, obj2):
            """A distance function between two TextBoxes.

            Consider the bounding rectangle for obj1 and obj2.
            Return its area less the areas of obj1 and obj2,
            shown as 'www' below. This value may be negative.
                    +------+..........+ (x1, y1)
                    | obj1 |wwwwwwwwww:
                    +------+www+------+
                    :wwwwwwwwww| obj2 |
            (x0, y0) +..........+------+
            """
            x0 = min(obj1.x0, obj2.x0)
            y0 = min(obj1.y0, obj2.y0)
            x1 = max(obj1.x1, obj2.x1)
            y1 = max(obj1.y1, obj2.y1)
            return ((x1-x0)*(y1-y0) - obj1.width*obj1.height - obj2.width*obj2.height)

        def isany(obj1, obj2):
            """Check if there's any other object between obj1 and obj2.
            """
            x0 = min(obj1.x0, obj2.x0)
            y0 = min(obj1.y0, obj2.y0)
            x1 = max(obj1.x1, obj2.x1)
            y1 = max(obj1.y1, obj2.y1)
            objs = set(plane.find((x0, y0, x1, y1)))
            return objs.difference((obj1, obj2))

        def key_obj(t):
            (c,d,_,_) = t
            return (c,d)

        # XXX this still takes O(n^2)  :(
        dists = []
        for i in range(len(boxes)):
            obj1 = boxes[i]
            for j in range(i+1, len(boxes)):
                obj2 = boxes[j]
                dists.append((0, dist(obj1, obj2), obj1, obj2))
        # We could use dists.sort(), but it would randomize the test result.
        dists = csort(dists, key=key_obj)
        plane = Plane(self.bbox)
        plane.extend(boxes)
        while dists:
            (c, d, obj1, obj2) = dists.pop(0)
            if c == 0 and isany(obj1, obj2):
                dists.append((1, d, obj1, obj2))
                continue
            if (isinstance(obj1, (LTTextBoxVertical, LTTextGroupTBRL)) or
                isinstance(obj2, (LTTextBoxVertical, LTTextGroupTBRL))):
                group = LTTextGroupTBRL([obj1, obj2])
            else:
                group = LTTextGroupLRTB([obj1, obj2])
            plane.remove(obj1)
            plane.remove(obj2)
            dists = [ (c,d,obj1,obj2) for (c,d,obj1,obj2) in dists
                      if (obj1 in plane and obj2 in plane) ]
            for other in plane:
                dists.append((0, dist(group, other), group, other))
            dists = csort(dists, key=key_obj)
            plane.add(group)
        assert len(plane) == 1, str(len(plane))
        return list(plane)

    def analyze(self, laparams):
        # textobjs is a list of LTChar objects, i.e.
        # it has all the individual characters in the page.
        (textobjs, otherobjs) = fsplit(lambda obj: isinstance(obj, LTChar), self)
        for obj in otherobjs:
            obj.analyze(laparams)
        if not textobjs:
            return
        textlines = list(self.group_objects(laparams, textobjs))
        self.create_tabstops(laparams, textlines)
        tables, remaining_textlines = self.create_tables(laparams, textlines, [rect for rect in otherobjs if isinstance(rect, LTRect)])
        (empties, remaining_textlines) = fsplit(lambda obj: obj.is_empty(), remaining_textlines)
        for obj in empties:
            obj.analyze(laparams)
        textboxes = list(self.group_textlines(laparams, remaining_textlines))
        if -1 <= laparams.boxes_flow and laparams.boxes_flow <= +1 and textboxes:
            self.groups = self.group_textboxes(laparams, textboxes)
            assigner = IndexAssigner()
            for group in self.groups:
                group.analyze(laparams)
                assigner.run(group)
            textboxes.sort(key=lambda box: box.index)
        else:
            def getkey(box):
                if isinstance(box, LTTextBoxVertical):
                    return (0, -box.x1, box.y0)
                else:
                    return (1, box.y0, box.x0)
            textboxes.sort(key=getkey)
        self._objs = textboxes + tables + otherobjs + empties
        return


##  LTFigure
##
class LTFigure(LTLayoutContainer):

    def __init__(self, name, bbox, matrix):
        self.name = name
        self.matrix = matrix
        (x, y, w, h) = bbox
        bbox = get_bound(apply_matrix_pt(matrix, (p, q))
                         for (p, q) in ((x, y), (x+w, y), (x, y+h), (x+w, y+h)))
        LTLayoutContainer.__init__(self, bbox)
        return

    def __repr__(self):
        return ('<%s(%s) %s matrix=%s>' %
                (self.__class__.__name__, self.name,
                 bbox2str(self.bbox), matrix2str(self.matrix)))

    def analyze(self, laparams):
        if not laparams.all_texts:
            return
        LTLayoutContainer.analyze(self, laparams)
        return


##  LTPage
##
class LTPage(LTLayoutContainer):
    tabstops = []

    def __init__(self, pageid, bbox, rotate=0):
        LTLayoutContainer.__init__(self, bbox)
        self.pageid = pageid
        self.rotate = rotate
        self.tabstops = []
        return

    def __repr__(self):
        return ('<%s(%r) %s rotate=%r>' %
                (self.__class__.__name__, self.pageid,
                 bbox2str(self.bbox), self.rotate))

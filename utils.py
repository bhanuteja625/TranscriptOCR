from paddleocr import PaddleOCR

from PIL import Image, ImageDraw

import numpy as np
import pandas as pd

import time

import math
from typing import Tuple, Union
from statistics import mode, mean

import re
from fuzzywuzzy import fuzz


class TranscriptOCR():

    def __init__(self):
        self.ocr = PaddleOCR(det_model_dir="./models/det_inference", rec_model_dir="./models/rec_inference",use_angle_cls=False, lang='en')

        self.target_words = ["course","title","grade","subject","credits","result","marks", "code", "total", "internal",'s.no']
    
    def __paddle_to_easy(self, result):
        results = []
        for r in result[0]:
            temp = []
            temp.append(r[0])
            temp.append(r[1][0])
            temp.append(r[1][1])
            results.append(temp)
        return results
    
    def __preprocess_text_objects(self, results):
        filtered = results
        transformed = []
        merged = []
        # print(results)
        for obj in filtered:
            center_x = sum([i[0] for i in obj[0]])//4
            center_y = sum([i[1] for i in obj[0]])//4
            height = max([abs(obj[0][0][1]-i[1]) for i in obj[0]])
            width = max([abs(obj[0][0][0]-i[0]) for i in obj[0]])
            transformed.append([[center_x,center_y,height,width],obj[1]])

        return transformed
    
    def __process_horizontal(self, results, size=1600):
        y = size/110
        horizontal=[]
        i=0

        unique_results = []
        seen_rectangles = []
        for rectangle, text in results:
            rectangle_key = tuple(rectangle)
            if rectangle_key not in seen_rectangles:
                unique_results.append([rectangle, text])
                seen_rectangles.append(rectangle_key)
        results = unique_results
        results=sorted(results, key=lambda x: (x[0][1], x[0][0]))

        while i<len(results):
            result=results[i]
            temp=[]
            k=0
            for j in results[i:]:
                if abs(result[0][1]-j[0][1])<=y:
                    temp.append(j)
                    k+=1
            i+=k
            temp=sorted(temp, key=lambda x:x[0])
            horizontal.append(temp)
        return horizontal
    
    def __find_text_bounding_box(self, data, target_words):
        bounding_boxes = []

        for item in data:
            text = item[1]
            bbox = item[0]
            text_lower = text.lower()

            for target_word in target_words:
                regex_pattern = re.compile(rf"\b\w{{0,{len(target_word)+1}}}{re.escape(target_word)}\w{{0,{len(target_word)+1}}}\b", re.IGNORECASE)
                if regex_pattern.search(text_lower):
                    bounding_boxes.append([bbox, text])
                    break
                if len(target_word) - 1 <= len(text_lower) <= len(target_word) + 1:
                    if fuzz.partial_ratio(target_word, text_lower) > 85:
                        bounding_boxes.append([bbox, text])
                        break

        return bounding_boxes
    
    def __search_headers(self, results, headers):
        idx_list = []
        for header in headers:
            for i in range(len(results)):
                result = results[i]
                lst = [res[1] for res in result]
                if header[1] in lst:
                    idx_list.append(i)
        idx = mode(idx_list)
        return idx,results[idx]
    
    def __is_overlapping(self,rect1, rect2):
        rect1_left = rect1[0] - rect1[3] / 2
        rect1_right = rect1[0] + rect1[3] / 2
        rect2_left = rect2[0] - rect2[3] / 2
        rect2_right = rect2[0] + rect2[3] / 2

        if rect1_left <= rect2_right and rect1_right >= rect2_left:
            return True, min(rect1_right-rect2_left,rect2_right-rect1_left)
        elif rect2_left <= rect1_right and rect2_right >= rect1_left:
            return True, min(rect1_right-rect2_left,rect2_right-rect1_left)
        elif rect1_left >= rect2_right and rect1_right >= rect2_left:
            return False, abs(rect2_right - rect1_left)
        elif rect2_left >= rect1_right and rect2_right >= rect1_left:
            return False, abs(rect1_right-rect2_left)
        else:
            return False, 1000
    
    def __header_overlap(self, results, header_boxes, tolerence=10):
        overlaps = []
        for header in header_boxes:
            overlaps.append([header])

            for result in results:
                header_t = header[0][1]-(header[0][2]/2)
                header_b = header[0][1]+(header[0][2]/2)
                result_t = result[0][1]-(result[0][2]/2)
                result_b = result[0][1]+(result[0][2]/2)
                if header_t-result_b<tolerence and result_t<header_b and self.__is_overlapping(header[0],result[0])[0] and self.__is_overlapping(header[0],result[0])[1]>20 and result!=header:
                    overlaps[-1].append(result)
                elif header_t<result_b and result_t-header_b<tolerence and self.__is_overlapping(header[0],result[0])[0] and self.__is_overlapping(header[0],result[0])[1]>20 and result!=header:
                    overlaps[-1].append(result)

        # print(overlaps)
        processed_headers = []
        for overlap in overlaps:
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')

            text = ""
            for rect, txt in overlap:
                center_x, center_y, height, width = rect
                x1 = center_x - width/2
                x2 = center_x + width/2
                y1 = center_y - height/2
                y2 = center_y + height/2

                min_x = min(min_x, x1)
                max_x = max(max_x, x2)
                min_y = min(min_y, y1)
                max_y = max(max_y, y2)
                text = text+" "+txt

            bounding_rect = [(min_x+max_x)/2, (min_y+max_y)/2, max_y - min_y, max_x - min_x]
            processed_headers.append([bounding_rect, text])

        seen_first_elements = []
        final_headers = []
        for sublist in processed_headers:
            first_element = sublist[0]
            if first_element not in seen_first_elements:
                final_headers.append(sublist)
                seen_first_elements.append(first_element)
        sorted_headers = sorted(final_headers, key=lambda x: x[0][0])
        return sorted_headers
    
    def __process_results(self,results, headers):
        def is_inside(rect1, rect2):
            x1, y1, h1, w1 = rect1
            x2, y2, h2, w2 = rect2
            return x1-(w1/2) >= x2-(w2/2) and y1-(h1/2) >= y2-(h2/2) and x1 + (w1/2) <= x2 + (w2/2) and y1 + (h1/2) <= y2 + (h2/2)

        final_results = []

        for result_rect, result_text in results:
            is_inside_header = False
            for header_rect, header_text in headers:
                if is_inside(result_rect, header_rect):
                    final_results.append([header_rect, header_text])
                    is_inside_header = True
                    break
            if not is_inside_header:
                final_results.append([result_rect, result_text])
        return final_results
        
    def __extract_text(self, img_path):

        image = Image.open(img_path)
        results = self.ocr.ocr(img_path)
        preprocessed = self.__preprocess_text_objects(self.__paddle_to_easy(results))
        preprocessed_x = self.__process_horizontal(preprocessed, image.size[1])
        header_boxes = self.__find_text_bounding_box(preprocessed, self.target_words)
        actual_headers = self.__search_headers(preprocessed_x, header_boxes)

        processed_headers = self.__header_overlap(preprocessed, actual_headers[1], image.size[1]/250)
        processed_results = self.__process_results(preprocessed, processed_headers)
        processed_horizontal = self.__process_horizontal(processed_results)

        return processed_results, processed_horizontal, processed_headers
    
    def __process_df(self, df):
        columns = list(df.columns)
        relations = []
        for i in range(len(columns)-1):
            test_cols = columns[i+1:]
            ref=columns[i]
            for test in test_cols:
                relationship_exists = all((df[ref] == df[test]) | (df[[ref, test]].isnull().any(axis=1)))
                if relationship_exists:
                    relations.append([ref,test])

        drop_cols=set()
        for relation in relations:
            name = ' '.join(relation)
            df[name] = df.apply(lambda row: row[relation[0]] if row[relation[0]] == row[relation[1]] else row[relation[0]] + row[relation[1]] if pd.notnull(row[relation[0]]) and pd.notnull(row[relation[1]]) else row[relation[0]] if pd.notnull(row[relation[0]]) else row[relation[1]], axis=1)
            i = min(columns.index(relation[0]), columns.index(relation[1]))
            columns.insert(i, name)
            df = df[columns]
            drop_cols.update(relation)
        df.drop(list(drop_cols), axis=1, inplace=True)
        return df
        
    def draw_boxes(self, img_path):
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        results,_,_ = self.__extract_text(img_path)
        for result in results:
            x_center, y_center, height, width = result[0]
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=2)
        return image
    
    def __extract_df(self, processed_horizontal, processed_headers):
        idx, headers = self.__search_headers(processed_horizontal, processed_headers)

        headers = sorted(headers, key=lambda x: x[0][0])

        header_names = [header[1] for header in headers]

        df = pd.DataFrame(columns=header_names)

        prev_y = mean([header[0][1] for header in headers])
        prev_dist = float('inf')

        analyze_text = processed_horizontal[idx+1:]

        data = []
        for text in analyze_text:
            y = mean([int(i[0][1]) for i in text])
            dist = y - prev_y
            if dist>=2*prev_dist:
                break
            temp = {}
            min_temp = {}
            assigned = []
            closest = {col[1]:[float("inf"), ""] for col in headers}
            for col in headers:
                for item in text:
                    if self.__is_overlapping(col[0],item[0])[0]:
                        temp[col[1]]=item[1]
                        closest[col[1]]=[0,item[1]]
                        assigned.append(item[1])
                        break
                    if self.__is_overlapping(col[0],item[0])[1]<closest[col[1]][0]:
                        closest[col[1]]=[self.__is_overlapping(col[0],item[0])[1],item[1]]

            min_values = {}
            for key, value in closest.items():
                if value[1] in min_values:
                    if value[0] < min_values[value[1]][0]:
                        min_values[value[1]] = (value[0], key)
                else:
                    min_values[value[1]] = (value[0], key)
            closest = {min_key: closest[min_key] for _, min_key in min_values.values()}

            for key in closest.keys():
                if closest[key][1] not in assigned:
                    temp[key]=closest[key][1]

            data.append(temp)
            prev_y = y
            prev_dist = dist


        dfs = [pd.DataFrame([row_data], columns=header_names) for row_data in data]
        df = pd.concat(dfs, ignore_index=True)
        df = self.__process_df(df)
        return df.to_dict()
    
    def __find_target_box(self, data, target_words, non_target_words=[]):
        bounding_boxes = []

        for item in data:
            text = item[1]
            bbox = item[0]
            text_lower = text.lower()

            non_target_present = any(re.search(rf"\b{re.escape(word)}\b", text_lower) for word in non_target_words)
            if non_target_present:
                continue
            for target_word in target_words:
                regex_pattern = re.compile(rf"\b\w{{0,{len(target_word)+1}}}{re.escape(target_word)}\w{{0,{len(target_word)+1}}}\b", re.IGNORECASE)
                if regex_pattern.search(text_lower):
                    bounding_boxes.append([bbox, text])
                    break
                if len(target_word) - 1 <= len(text_lower) <= len(target_word) + 1:
                    if fuzz.partial_ratio(target_word, text_lower) > 85:
                        bounding_boxes.append([bbox, text])
                        break
        return bounding_boxes

    def __find_next(self, horizontal, target):
        for line in horizontal:
            if target in line:
                if line.index(target)+1<len(line):
                    return line[line.index(target)+1]
        return ""
    
    def __get_fields(self, preprocessed_x, processed_results):

        target_name = ["name", "student", "candidate"]
        non_target_name = ["course","father","mother"]

        target_university = ['university',"college",'education','institute','foundation']

        target_course = ['computer', 'mechanical','civil',"electronics", "electrical",'department',"artificial","intelligence","engineering","information","technology"]
        non_target_course = ['institute','college','university']

        target_gpa=['cgpa','sgpa','gpa']
        non_target_gpa = []



        name_field = self.__find_target_box(processed_results, target_name, non_target_name)

        if len(name_field)==0:
            name=""
        else:
            name = name_field[0][1]
            if len(name.split(" "))>4 or ':' in name:
                name_list=name.split(" ")
                id = list([0])
                for target in target_name:
                    i = name.find(target)
                    if i>-1:
                        id.append(i+len(target))
                name = name[max(id):]

            else:
                name = self.__find_next(preprocessed_x,name_field[0])[1]

        university= self.__find_target_box(processed_results, target_university)

        if len(university)==0:
            university=""
        else:
            university=university[0][1]

        branch= self.__find_target_box(processed_results, target_course, non_target_course)
        if len(branch)==0:
            branch=""
        else:
            branch=branch[0][1]

        gpa_fields = self.__find_target_box(processed_results, target_gpa)
        gpa_fields = [txt for txt in gpa_fields if len(txt[1].split(" "))<3]
        gpas = {}
        if len(gpa_fields)>0:
            for i in gpa_fields:
                gpas[i[1]] = self.__find_next(preprocessed_x, i)

        return name, university, branch, gpas
    
    def extract_json(self, img_path):
        processed_results, processed_horizontal, processed_headers = self.__extract_text(img_path)
        course_data = self.__extract_df(processed_horizontal, processed_headers)
        name, university, branch, gpas = self.__get_fields(processed_horizontal, processed_results)
        json_data = {
            "name":name,
            "university":university,
            "course":branch,
            "course table":course_data,
            "gpas":gpas
        }
        return json_data
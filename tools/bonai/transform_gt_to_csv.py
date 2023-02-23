import json
import pandas as pd


def save_csv(fileName, saveDict):
    df = pd.DataFrame(saveDict)
    df.to_csv(fileName, index=False, header=True)


def process_points(points):
    # points = points[0]
    start = "{} {},".format(points[0], points[1])
    end = "{} {}".format(points[0], points[1])
    result_str = ""
    for i in range(len(points)):
        if i % 2 == 0:
            result_str += str(points[i])
            result_str += " "
        else:
            result_str += str(points[i])
            result_str += ","
    result_str = result_str + start + start + end
    result_str = "(({}))".format(result_str)
    return result_str


gt_data_path = "./bonai_shanghai_xian_test_roof.json"
json_file = open(gt_data_path, "r")
content = json_file.read()
json_content = json.loads(content)
annotations = json_content["annotations"]
images = json_content["images"]
images_ids = []
images_filenames = []
results = []
for item in images:
    images_ids.append(item["id"])
    images_filenames.append(item["file_name"])
id_2_filename = dict(zip(images_ids, images_filenames))
for image_id in images_ids:
    building_index = 0
    for ann in annotations:
        if ann["area"] > 500 and ann["image_id"] == image_id:
            points = ann["roof_mask"]
            points = process_points(points)
            result = dict(
                ImageId=id_2_filename[image_id],
                BuildingId=building_index,
                PolygonWKT_Pix=f"POLYGON {points}",
                Confidence=1,
            )
            results.append(result)
            building_index += 1

save_csv("./test.csv", results)
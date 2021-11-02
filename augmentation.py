import os
import numpy as np
import cv2
import json
from collections import defaultdict
from tqdm import tqdm


dataset='val'
angle='90'
# merge = False

datasets = ['val', 'train']
angles = ['90', '270']

coco_path = os.path.join(os.getcwd(), 'datasets')
ann_path = os.path.join(coco_path, 'annotations')
server_path = '/mnt/data/COCO/coco2017'
json_path = os.path.join(coco_path, 'rotate_{}2017_sep.json'.format(dataset))


# json_path = os.path.join(coco_path, 'rotate_val2017.json')
# file = open(json_path, "r")
# jsonString = json.load(file)



def rotate_coco(dataset='val', angle='90', coco_path=coco_path, ann_path=ann_path):
    # if dataset=='val':
    #     dataset_path = os.path.join(coco_path, '{}2017'.format(dataset))
    # elif dataset=='train':
    #     dataset_path = os.path.join(server_path, '{}2017'.format(dataset))

    dataset_path = os.path.join(server_path, '{}2017'.format(dataset))

    json_path = os.path.join(ann_path, 'person_keypoints_{}2017.json'.format(dataset, angle))

    rotate_path = os.path.join(coco_path, '{}2017_rotate_data'.format(dataset))

    file = open(json_path, "r")
    jsonString = json.load(file)

    ## remove RLE segmentations
    imgToAnns = defaultdict(list)
    ids_ann = []
    ids1 = []
    for i, ann in enumerate(jsonString['annotations']):
        if ann['iscrowd'] > 0:
            # jsonString['annotations'].pop(i)
            ids1.append(i)
            ids_ann.append(ann['image_id'])
        else:
            imgToAnns[ann['image_id']].append(ann)

    ids1.reverse()
    for i in ids1:
        jsonString['annotations'].pop(i)


    ids2 = []
    for i, image in enumerate(jsonString['images']):
        if image['id'] in ids_ann:
            ids2.append(i)
            # jsonString['images'].pop(i)

    ids2.reverse()
    for i in ids2:
        jsonString['images'].pop(i)


    ids = []
    ids_img = []
    for idx, img_info in tqdm(enumerate(jsonString['images'])):

        file_name = img_info["file_name"]
        img_path = os.path.join(dataset_path, file_name)
        img = cv2.imread(img_path)

        image_id = img_info['id']

        if image_id not in imgToAnns:
            ids.append(idx)
            # ids_img.append(image_id)
            continue

        anns = imgToAnns[image_id]
        # coco.showAnns(anns)

        height = img_info['height']
        width = img_info['width']


        for ann_idx, ann in enumerate(anns):
            # annId = ann['id']
            ## rotate segmentations
            segs = ann["segmentation"]
            for seg_idx in range(len(segs)):
                seg = np.array(segs[seg_idx])
                seg = np.fliplr(seg.reshape((-1, 2)))

                if angle=='90':
                    seg[:, ::2] = height - seg[:, ::2]
                elif angle=='270':
                    seg[:, 1::2] = width - seg[:, 1::2]
                seg = seg.reshape(-1)
                seg = [float(num) for num in seg]
                segs[seg_idx] = list(seg)

            # segs = np.round(segs)
            ann["segmentation"] = segs


            ## rotate keypoints
            kp = ann["keypoints"]
            kp = np.array(kp).reshape((-1, 3))
            kp = np.hstack((kp[:, 1].reshape(-1, 1), kp[:, 0].reshape(-1, 1), kp[:, 2].reshape(-1, 1)))
            kp_idxs = []
            for kp_idx in range(len(kp)):
                if (kp[kp_idx, 2] != 0):
                    kp_idxs.append(kp_idx)

            if angle=='90':
                kp[kp_idxs, 0] = height - kp[kp_idxs, 0]
            elif angle=='270':
                kp[kp_idxs, 1] = width - kp[kp_idxs, 1]

            kp = kp.reshape(-1)
            kp = list([int(x) for x in kp])

            ann["keypoints"] = kp


            ## rotate bbox
            bbox = ann["bbox"]

            if angle=='90':
                rotate_bbox = [height - bbox[1] - bbox[3], bbox[0], bbox[3], bbox[2]]
            elif angle=='270':
                rotate_bbox = [bbox[1], width - bbox[0] - bbox[2], bbox[3], bbox[2]]


            ann["bbox"] = rotate_bbox

            ## modify ann id
            # ann["category_id"] = 2
            if angle=='90':
                ann["id"] = 100000000 + ann["id"]
                ann["image_id"] = 10000000 + img_info['id']
                ann["category_id"] = 2
            elif angle=='270':
                ann["id"] = 200000000 + ann["id"]
                ann["image_id"] = 20000000 + img_info['id']
                ann["category_id"] = 3


        # modify img
        if angle=='90':
            img_info['id'] = 10000000 + img_info['id']
        elif angle=='270':
            img_info['id'] = 20000000 + img_info['id']

        img_info['height'], img_info['width'] = width, height
        file_name = angle + '_' + file_name
        img_info['file_name'] = file_name

        # if angle=='90':
        #     rotate_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # elif angle=='270':
        #     rotate_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #
        # cv2.imwrite(os.path.join(rotate_path, file_name), rotate_img)

    if angle == '90':
        jsonString['categories'][0]['id'] = 2
        jsonString['categories'][0]['name'] = '90'
    elif angle == '270':
        jsonString['categories'][0]['id'] = 3
        jsonString['categories'][0]['name'] = '270'

    # jsonString['categories'][0]['id'] = 2
    # jsonString['categories'][0]['name'] = 'fallen'

    ids.reverse()
    for i in ids:
        jsonString['images'].pop(i)

    json_output = os.path.join(ann_path, 'rotate_{}_{}2017_.json'.format(angle, dataset))

    with open(json_output, 'w') as f:
        output = json.dumps(jsonString)
        f.write(output)

######

for dataset in datasets:
    for angle in angles:
        rotate_coco(dataset=dataset, angle=angle, coco_path=coco_path, ann_path=ann_path)





# ## id change
# datasets = ['train', 'val']
#
#
# json_path = os.path.join(coco_path, 'rotate_train2017.json')
# file = open(json_path, "r")
# jsonString = json.load(file)
# for ann in jsonString['annotations']:
#     ann['categoriy_id'] = 2
#
# jsonString['categories']['id'] = 2






###########################
# load and display image
# import io
# I = io.imread(os.path.join(folder_path, img['file_name']))
# plt.axis('off')
# plt.imshow(I)
# plt.show()
#
# # load and display instance annotations
# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
#
# plt.close()
#
# def test(json_path):
#     pylab.rcParams['figure.figsize'] = (8.0, 10.0)
#     coco=COCO(json_path)
#
#     cats = coco.loadCats(coco.getCatIds())
#     nms=[cat['name'] for cat in cats]
#     print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
#     nms = set([cat['supercategory'] for cat in cats])
#     print('COCO supercategories: \n{}'.format(' '.join(nms)))
#
#     catIds = coco.getCatIds(catNms=['person'])
#     imgIds = coco.getImgIds(catIds=catIds )
#     img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#     print(img)
#
#     I = io.imread(img['coco_url'])
#     plt.imshow(I)
#     plt.axis('off')
#     annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
#     anns = coco.loadAnns(annIds)
#     coco.showAnns(anns)
#     plt.show(block=False)
#     # plt.pause(5)
#     plt.close()
#
# for _ in range(100):
#     test(json_path=json_path)






# listdir1 = os.listdir('datasets/val2017_rotate_data')
# listdir1 = ['datasets/val2017_rotate_data/'+name for name in listdir1]
# listdir2 = os.listdir('datasets/train2017_rotate_data')
# listdir2 = ['datasets/train2017_rotate_data/'+name for name in listdir2]
# listdir = listdir1+listdir2
#
# for img_path in listdir:
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(img_path, img)


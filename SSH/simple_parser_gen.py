import cv2
import numpy as np
import scipy.io as spio
import traceback

def get_all_imgs(all_imgs, mat, path, classes_count, imageset):
	for i in range(len(mat['event_list'])):
		event = mat['event_list'][i]
		print (event)
		cnt = 0
		try:
			for face_list in mat['file_list'][i]:
				filename = path+event+'/'+face_list+'.jpg'
				class_name ='face'
				if filename not in all_imgs:
					# if(len(mat['face_bbx_list'][i][cnt]) == 0):
					if(np.isnan(mat['face_bbx_list'][i][cnt]).any()):
						print('skip')
						continue
					all_imgs[filename] = {}
					img = cv2.imread(filename)
					(rows,cols) = img.shape[:2]
					all_imgs[filename]['filepath'] = filename
					all_imgs[filename]['width'] = cols
					all_imgs[filename]['height'] = rows
					all_imgs[filename]['bboxes'] = []
					all_imgs[filename]['imageset'] = imageset
				
				try:
					for bb in mat['face_bbx_list'][i][cnt]:
						classes_count[class_name] += 1
						all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(bb[0]), 'y1': int(bb[1]), 'x2': int(bb[2]+bb[0]), 'y2': int(bb[3]+bb[1])})
				except:
					bb = mat['face_bbx_list'][i][cnt]
					classes_count[class_name] += 1
					all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(bb[0]), 'y1': int(bb[1]), 'x2': int(bb[2]+bb[0]), 'y2': int(bb[3]+bb[1])})
				cnt+=1
				break
			break
		except:
			print(filename)
			print(traceback.format_exc())
			continue
	return all_imgs



def get_data(input_path):
	found_bg = False
	all_imgs = {}

	classes_count = {'face':0}

	class_mapping = {'face':0}

	visualise = True
	

	print('Parsing annotation files')

	trainImagePath = 'WIDER/WIDER_train/images/'
	trainAnnotations = 'WIDER/wider_face_split/wider_face_train.mat'

	valImagePath = 'WIDER/WIDER_val/images/'
	valAnnotations = 'WIDER/wider_face_split/wider_face_val.mat'

	mat = spio.loadmat(trainAnnotations, squeeze_me=True)
	all_imgs = get_all_imgs(all_imgs, mat, trainImagePath, classes_count, 'train')
	mat = spio.loadmat(valAnnotations, squeeze_me=True)
	all_imgs = get_all_imgs(all_imgs, mat, valImagePath, classes_count, 'val')

	all_data = []
	for key in all_imgs:
		# print(key)
		# print(all_imgs[key])
		all_data.append(all_imgs[key])
	
	return all_data, classes_count, class_mapping


# all_data, classes_count, class_mapping = get_data('')

# print (len(all_data))
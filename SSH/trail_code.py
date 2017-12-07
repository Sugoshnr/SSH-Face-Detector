def f_measure(gt_bboxes, pred_bboxes):
    deteval = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype='int')
    for gt_idx in range(deteval.shape[0]):
        for pred_idx in range(deteval.shape[1]):
            gt_bbox = gt_bboxes[gt_idx]
            pred_bbox = pred_bboxes[pred_idx]
            curr_iou = data_generators.iou(gt_bbox, pred_bbox)
            deteval[gt_idx, pred_idx] = 1 if curr_iou >= C.char_eval_min_iou else 0
    r_count = np.sum(np.sum(deteval, axis=1) > 0)
    r = r_count / float(len(gt_bboxes))
    p_count = np.sum(np.sum(deteval, axis=0) > 0)
    p = p_count / float(len(pred_bboxes))
    f = 0 if r == 0 or p == 0 else (2. * r * p) / (r + p)
    return r_count, r, p_count, p, f

def f_measure_rpn(C, rpn_bboxes, rpn_probs):
    gt_bboxes = np.asarray([[bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
                            for bbox in C.img_data['bboxes_scaled_output']], dtype='int')
    # print(rpn_bboxes, gt_bboxes, rpn_probs)
    pred_bboxes = np.asarray([rpn_bboxes[idx, :] for idx in range(len(rpn_bboxes)) if rpn_probs[idx] >= 0.5])
    r_count, r, p_count, p, f = f_measure(gt_bboxes, pred_bboxes)
    print('''
          RPN recall: {} = {}/{},
          RPN precision: {} = {}/{},
          RPN f-measure: {}'''.format(r, r_count, len(gt_bboxes), p, p_count, len(pred_bboxes), f))
    C.eval['rpn_rcount'] = [r_count] if 'rpn_rcount' not in C.eval else C.eval['rpn_rcount'] + [r_count]
    C.eval['rpn_recall'] = [r] if 'rpn_recall' not in C.eval else C.eval['rpn_recall'] + [r]
    C.eval['rpn_pcount'] = [p_count] if 'rpn_pcount' not in C.eval else C.eval['rpn_pcount'] + [p_count]
    C.eval['rpn_precision'] = [r] if 'rpn_precision' not in C.eval else C.eval['rpn_precision'] + [r]
    C.eval['rpn_tcount'] = [len(gt_bboxes)] if 'rpn_tcount' not in C.eval else C.eval['rpn_tcount'] + [len(gt_bboxes)]
    C.eval['rpn_ocount'] = [len(pred_bboxes)] if 'rpn_ocount' not in C.eval else C.eval['rpn_ocount'] + [len(pred_bboxes)]

    if C.diagnose:
        img = cv2.imread(C.img_data['filepath'])
        (width, height) = (C.img_data['width'], C.img_data['height'])
        (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)
        img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
        num_bboxes = len(C.img_data['bboxes'])
        for bbox_num in range(num_bboxes):
            bbox = C.img_data['bboxes_scaled_input'][bbox_num]
            x1_gt, x2_gt, y1_gt, y2_gt = map(int, [bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']])
            cv2.rectangle(img, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 255, 0), 1)
        num_bboxes = len(pred_bboxes)
        for bbox_num in range(num_bboxes):
            x1, y1, x2, y2 = map(lambda x: x * C.rpn_stride, pred_bboxes[bbox_num].astype('int'))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        p_ = C.img_data['filepath'].split(os.sep)[:-2]
        f_ = C.img_data['filepath'].split(os.sep)[-1]
        f_ = '_rpn.'.join(f_.split('.'))
        cv2.imwrite(os.path.join(os.sep.join(p_), 'diagnose', f_), img)
        # cv2.imshow('gt boxes and anchors {}'.format(f), img)
        # cv2.waitKey(0)
    return f, r, p
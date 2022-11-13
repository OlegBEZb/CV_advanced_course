import numpy as np
from matplotlib import pyplot as plt
import cv2


def keypoint_matcher(img1, img2, n_points=8,
                     filter_neighbours=True,
                     draw_matches=False):
    """
    :param img1:
    :param img2:
    :param n_points: the number of matched points to keep finally. -1 means to keep all. 8 by default
    :param filter_neighbours: as per Lowe's paper
    :param draw_matches:
    :return:
    """

    # another cool detector https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
    # Initiate ORB detector
    #     orb = cv.ORB_create()

    # may return duplicated points with different descriptions
    # Also, any peaks above 80% of the highest peak are converted into a new keypoint.
    # This new keypoint has the same location and scale as the original.
    # But it’s orientation is equal to the other peak.
    sift = cv2.SIFT_create(nfeatures=n_points)
    kp1, descriptors1 = sift.detectAndCompute(img1, None)
    kp2, descriptors2 = sift.detectAndCompute(img2, None)

    # TODO: if works slow, replace with KD
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)
    # Once it is created, two important methods are BFMatcher.match() and BFMatcher.knnMatch(). First one returns the
    # best match. Second method returns k best matches where k is specified by the user. It may be useful when we need
    # to do additional work on that.
    # TODO: check why contains duplicated pairs. It doesn't affect chaining
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    if filter_neighbours:
        # in some cases, the second closest-match may be very near to the first. It may happen due to noise or some
        # other reasons. In that case, ratio of closest-distance to second-closest distance is taken. If it is greater
        # than 0.8, they are rejected. It eliminates around 90% of false matches while discards only 5% correct matches,
        # as per Lowe's paper.
        len_before = len(matches)
        # With lower thresholds it's even better
        matches = [m for m in matches if m[0].distance / m[1].distance < 0.8]
        print(f'Before filtering neighbours: {len_before}. After: {len(matches)}')

    if draw_matches:
        # cv2.drawMatches() draws the matches. It stacks two images horizontally and draw lines from first image to
        # second image showing best matches. There is also cv2.drawMatchesKnn which draws all the k best matches. If k=2,
        # it will draw two match-lines for each keypoint.
        plt.figure(figsize=(15, 15))
        show_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2,
                                          #                                          [(m[0],) for m in matches[:8]],  # the nearest neighbour of 8 matches
                                          [(m[0],) for m in matches],  # the nearest neighbour of 8 matches
                                          None)
        plt.imshow(show_matches)

    matches = sorted(matches, key=lambda x: x[0].distance)

    # extracting coordinates of matches points
    matched_points1 = []
    matched_points2 = []
    for m in matches:
        # according to the doc, queryIdx refers to the first keypoints and trainIdx refers to second keypoints
        # here we just take the closest point from all neighbours
        point_1 = kp1[m[0].queryIdx].pt
        point_2 = kp2[m[0].trainIdx].pt

        if point_1 not in matched_points1 and point_2 not in matched_points2:
            matched_points1.append(point_1)
            matched_points2.append(point_2)
        else:
            #             print('point 1 or point 2 are already in the list but this another direction')
            #             print(f'p1:{point_1}, p2:{point_2}')
            #             print(f'matched_points1:{matched_points1}')
            #             print(f'matched_points2:{matched_points2}')
            pass

    print("Some matches contained the same points. After deduplication:", len(matched_points1))
    assert len(matched_points1) >= 3, f'not enough points for affine transformation. Have {len(matched_points1)}'

    return matches, matched_points1, matched_points2, kp1, kp2


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    # https://github.com/auxiliary/imdist/blob/77a20c1f968e3678f7ba00919d89655bc1217195/matcher.py#L73
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, center=(int(x1), int(y1)), radius=3, color=(255, 0, 0), thickness=2)
        cv2.circle(out, center=(int(x2) + cols1, int(y2)), radius=3, color=(255, 0, 0), thickness=2)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 2)

    return out


def align_image(img_to_transform, matched_points1, matched_points2, method='affine'):
    assert len(matched_points1) == len(matched_points2)

    rows, cols, ch = img_to_transform.shape

    if method == 'affine':
        assert len(matched_points1) > 3, 'The number of points needed for getAffineTransform is 3'
        assert len(matched_points2) > 3, 'The number of points needed for getAffineTransform is 3'

        M = cv2.getAffineTransform(np.float32(matched_points1[:3]),
                                   np.float32(matched_points2[:3]))
        img_transformed = cv2.warpAffine(img_to_transform, M, (cols, rows))
    elif method == 'perspective':
        M = cv2.getPerspectiveTransform(np.float32(matched_points1[:4]),
                                        np.float32(matched_points2[:4]))
        img_transformed = cv2.warpPerspective(img_to_transform, M, (cols, rows))
    elif method == 'homography':
        M, _ = cv2.findHomography(np.float32(matched_points1),
                                  np.float32(matched_points2), cv2.RANSAC, 5.0)
        img_transformed = cv2.warpPerspective(img_to_transform, M, (cols, rows))
    else:
        raise ValueError("'method' has to be 'affine', 'perspective', or 'homography'")

    return img_transformed


def pansharpen(rgb_img, bw_img, align_method='affine', pansharp_method='hsv', **kp_matcher_args):
    print('kp_matcher_args', kp_matcher_args)
    assert len(rgb_img.shape) == 3, "Coloured image 'rgb_img' have to be in RGB"
    assert len(bw_img.shape) == 2, "Black-and-white image 'bw_img' has to be a 2-d array"

    if rgb_img.shape[:2] != bw_img.shape:
        print(f'Images are of different sizes {rgb_img.shape[:2]} vs {bw_img.shape} -> resizing')
        rows, cols = bw_img.shape
        rgb_img_res = cv2.resize(src=rgb_img, dst=None, dsize=(cols, rows),
                                 interpolation=cv2.INTER_CUBIC)
    else:
        rgb_img_res = rgb_img

    matches, matched_points1, matched_points2, kp1, kp2 = keypoint_matcher(
        rgb_img_res, bw_img, **kp_matcher_args)

    rgb_img_res = align_image(img_to_transform=rgb_img_res, matched_points1=matched_points1,
                              matched_points2=matched_points2, method=align_method)

    rgb_img_res_sharp = _pansharpen(rgb_img_res, bw_img, method=pansharp_method)

    return rgb_img_res_sharp, bw_img


def _pansharpen(rgb_im, bw_im, method='hsv', W=0.1):
    rgb_new = rgb_im.copy()

    R = rgb_im[:, :, 0]
    G = rgb_im[:, :, 1]
    B = rgb_im[:, :, 2]

    if method == 'simple_mean':
        # overflow encountered in ubyte_scalars
        bw_im_half = 0.5 * bw_im
        rgb_new[:, :, 0] = 0.5 * R + bw_im_half
        rgb_new[:, :, 1] = 0.5 * G + bw_im_half
        rgb_new[:, :, 2] = 0.5 * B + bw_im_half
    elif method == 'hsv':
        hsv_new = cv2.cvtColor(rgb_new, cv2.COLOR_RGB2HSV)
        hsv_new[:, :, 2] = bw_im
        rgb_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
    else:
        raise ValueError('wrong method')

    return rgb_new

from scipy.spatial import ConvexHull
import numpy as np


def normalize_kp(kp_source_value, kp_source_jacobian,
                 kp_driving_value, kp_driving_jacobian,
                 kp_driving_initial_value, kp_driving_initial_jacobian):
    source_area = ConvexHull(kp_source_value[0]).volume
    driving_area = ConvexHull(kp_driving_initial_value[0]).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_value_diff = (kp_driving_value - kp_driving_initial_value)
    kp_value_diff *= adapt_movement_scale
    kp_new_value = kp_value_diff + kp_source_value
    jacobian_diff = np.matmul(kp_driving_jacobian, np.expand_dims(np.linalg.inv(kp_driving_initial_jacobian[0]), 0))
    kp_new_jacobian = np.matmul(jacobian_diff, kp_source_jacobian)

    return kp_new_value, kp_new_jacobian

# def normalize_kp(kp_source_value, kp_source_jacobian,
#                  kp_driving_value, kp_driving_jacobian,
#                  kp_driving_initial_value, kp_driving_initial_jacobian,
#                  adapt_movement_scale=True,
#                  use_relative_movement=True, use_relative_jacobian=True):
#     if adapt_movement_scale:
#         tmp1 = np.array(kp_source_value[0].data.cpu().numpy())
#         tmp2 = np.array(kp_driving_initial_value[0].data.cpu().numpy())
#         source_area = ConvexHull(tmp1).volume
#         driving_area = ConvexHull(tmp2).volume
#         adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
#     else:
#         adapt_movement_scale = 1
#
#     kp_new_value = kp_driving_value
#     kp_new_jacobian = kp_driving_jacobian
#
#     if use_relative_movement:
#         kp_value_diff = (kp_driving_value - kp_driving_initial_value)
#         kp_value_diff *= adapt_movement_scale
#         kp_new_value = kp_value_diff + kp_source_value
#
#         if use_relative_jacobian:
#             jacobian_diff = torch.matmul(kp_driving_jacobian, torch.inverse(kp_driving_initial_jacobian))
#             kp_new_jacobian = torch.matmul(jacobian_diff, kp_source_jacobian)
#
#     return kp_new_value, kp_new_jacobian


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

LAYING_RAW_DATA_FILE = 'laying_grasps.csv'
STANDING_RAW_DATA_FILE = 'standing_grasps.csv'

NORM = 'stdnorm' # 'stdnorm', 'featurescaling'

TYPE = 'known' # 'whole', 'known', 'unknown'
FILL_STRATEGY = 'cero2mean' # 'cero2lesscontact' 'cero2mean'

# TODO
LABELS_OUT_FILE = 'labels-' + TYPE + '-t' + '-' + FILL_STRATEGY + '-' \
                + NORM + '.npy'
IMAGES_OUT_FILE = 'images-' + TYPE + '-t' + '-' + FILL_STRATEGY + '-' \
                + NORM + '.npy'

TACTILE_IMAGE_ROWS = 12
TACTILE_IMAGE_COLS = 11
ELECTRODES_INDEX_ROWS = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 3, 5])
ELECTRODES_INDEX_COLS = np.array([1, 2, 0, 1, 3, 0, 1, 4, 2, 1, 9, 8, 10, 9, 7, 10, 9, 6, 8, 9, 5, 4, 6, 5])

# TODO
laying_raw_df = pd.read_csv(LAYING_RAW_DATA_FILE)
standing_raw_df = pd.read_csv(STANDING_RAW_DATA_FILE)
whole_raw_df = pd.concat([laying_raw_df, standing_raw_df])

whole_raw_df.describe()



# Returns the values of the 8 neighbours of a given cell.
# This method is meant to be called with the gaps in tactile cells.
def get_neighbours(tactile_image, cell_x, cell_y):
    pad = 2
    padded_x = cell_x + pad
    padded_y = cell_y + pad
    
    padded = np.pad(tactile_image, ((pad, pad), (pad, pad)), 'constant') #0s
    
    neighbours_xs = [padded_x - 1, padded_x - 1, padded_x - 1, 
                     padded_x, padded_x, 
                     padded_x + 1, padded_x + 1, padded_x + 1]
    neighbours_ys = [padded_y - 1, padded_y, padded_y + 1,
                     padded_y - 1, padded_y + 1,
                     padded_y - 1, padded_y, padded_y + 1]
    num_neighbours = len(neighbours_xs)
    neighbours = []
    
    for i in range(num_neighbours):
        some_x = neighbours_xs[i]
        some_y = neighbours_ys[i]
        neighbours.append(padded[some_x, some_y])

    return neighbours

def ceros_2_mean(tactile_image):
    prev_tactile_image = np.copy(tactile_image)
    cero_xs, cero_ys = np.where(tactile_image == 0)

    for i in range(len(cero_xs)):
        cell_x = cero_xs[i]
        cell_y = cero_ys[i]
        cell_neighs = get_neighbours(prev_tactile_image, cell_x, cell_y)
        cell_neighs = [value for value in cell_neighs if value > 0.0]

        if len(cell_neighs) > 0:
            tactile_image[cell_x, cell_y] = np.mean(cell_neighs)
            
    return tactile_image    

def create_finger_tactile_image(finger_biotac, normalization, fill_strategy=1):
    tactile_image = np.zeros(shape=(TACTILE_IMAGE_ROWS, TACTILE_IMAGE_COLS))
    tactile_image[ELECTRODES_INDEX_ROWS, ELECTRODES_INDEX_COLS] = finger_biotac
    
    if fill_strategy == 'cero2lesscontact':
        # Strategy 1 - Fill with less contacted value
        # The maximum value corresponds to the less contacted electrode
        max_value = np.max(finger_biotac)
        tactile_image[tactile_image == 0] = max_value
    elif fill_strategy == 'cero2mean':
        # Strategy 2 - Fill with neighbours average
        tactile_image = ceros_2_mean(tactile_image)
        
        # Repeat in case that there were cells with no values as neighbours, they will now
        if np.min(tactile_image) == 0.0:
            tactile_image = ceros_2_mean(tactile_image)
    
    if normalization == 'stdnorm':
        tactile_image = (tactile_image - np.mean(tactile_image)) / (np.std(tactile_image))
    elif normalization == 'featurescaling':
        tactile_image = (tactile_image - np.min(tactile_image)) / (np.max(tactile_image) - np.min(tactile_image))
    
    return tactile_image


# FOR CREATING DATASETS
labels = whole_raw_df['slipped'].values
tactiles_df = whole_raw_df[['ff_biotac_1', 'ff_biotac_2', 'ff_biotac_3', 'ff_biotac_4', 'ff_biotac_5', 
                   'ff_biotac_6', 'ff_biotac_7', 'ff_biotac_8', 'ff_biotac_9', 'ff_biotac_10', 'ff_biotac_11', 
                   'ff_biotac_12', 'ff_biotac_13', 'ff_biotac_14', 'ff_biotac_15', 'ff_biotac_16', 'ff_biotac_17', 
                   'ff_biotac_18', 'ff_biotac_19', 'ff_biotac_20', 'ff_biotac_21', 'ff_biotac_22', 'ff_biotac_23', 
                   'ff_biotac_24']]

tactile_images = np.zeros(shape=(tactiles_df.shape[0], 1, TACTILE_IMAGE_ROWS, TACTILE_IMAGE_COLS))

for sample in range(tactiles_df.shape[0]):
    one_grasp = tactiles_df.iloc[sample].values
    tactile_images[sample] = create_finger_tactile_image(one_grasp[0:24], normalization=NORM, fill_strategy=FILL_STRATEGY)
    
some_grasp = 0
print(labels[some_grasp])
print(tactile_images[some_grasp])

whole_labels = labels
whole_images = tactile_images

np.save(LABELS_OUT_FILE, arr=whole_labels)
np.save(IMAGES_OUT_FILE, arr=whole_images)

print(whole_labels.shape)
print(whole_images.shape)

unique = int(whole_images.shape[0] / 5)

print(unique)

index = 2063

plt.figure()
plt.imshow(whole_images[0 * unique + index, 0, :, :], interpolation='nearest', )
plt.axis('off')
plt.title('original')
plt.colorbar()


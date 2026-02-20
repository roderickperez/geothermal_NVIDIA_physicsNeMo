import numpy as np

def test_idx():
    # Simulate the array handed to imshow (layer_data)
    # Shape [Y, X]
    layer_data = np.zeros((32, 32)) 
    layer_data[5, 10] = 100 # Max point (yellow, Injector) >> row 5, col 10
    layer_data[25, 3] = -100 # Min point (purple, Producer) >> row 25, col 3

    # To plot correctly with scatter(x, y):
    # x is column (dimension 1), y is row (dimension 0)

    inj_idx = np.unravel_index(np.argmax(layer_data), layer_data.shape)
    prod_idx = np.unravel_index(np.argmin(layer_data), layer_data.shape)

    print(f"Injector unravel_index (Y, X): {inj_idx}")
    print(f"Producer unravel_index (Y, X): {prod_idx}")
    
    # For scatter, we must flip them:
    scatter_inj_x = inj_idx[1]
    scatter_inj_y = inj_idx[0]
    
    print(f"Scatter Inj (X, Y): ({scatter_inj_x}, {scatter_inj_y})")

test_idx()

import numpy as np

def noisify_with_P(y_train, nb_classes, noise, random_state = None):

    if noise > 0.0:
        P = noise / (nb_classes - 1) * np.ones((nb_classes, nb_classes))
        np.fill_diagonal(P, (1 - noise) * np.ones(nb_classes))
        y_train_noisy = multiclass_noisify(y_train, P = P, random_state = random_state)

        actual_noise = (y_train_noisy != y_train).mean()

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P

def noisify_cifar10_asymmetric(y_train, noise, random_state = None):

    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # automobile <-> truck
        P[9, 9], P[9, 1] = 1. - n, n
        P[1, 1], P[1, 9] = 1. - n, n

        # bird <-> airplane
        P[2, 2], P[2, 0] = 1. - n, n
        P[0, 0], P[0, 2] = 1. - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1. - n, n
        P[5, 5], P[5, 3] = 1. - n, n

        # deer <-> horse
        P[4, 4], P[4, 7] = 1. - n, n
        P[7, 7], P[7, 4] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P = P, random_state = random_state)

        actual_noise = (y_train_noisy != y_train).mean()

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P

def noisify_cifar100_asymmetric(y_train, noise, random_state=None):
    
    nb_classes = 100
    P = np.eye(nb_classes)
    nb_superclasses = 20
    nb_subclasses = 5

    if noise > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i + 1) * nb_subclasses
            P[init:end, init:end] = build_p_for_cifar100(nb_subclasses, noise)

        y_train_noisy = multiclass_noisify(y_train, P = P, random_state = random_state)

        actual_noise = (y_train_noisy != y_train).mean()

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P

def build_p_for_cifar100(size, noise):
    
    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i + 1] = noise
    P[size - 1, 0] = noise

    return P

def multiclass_noisify(y, P, random_state = 0):
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y
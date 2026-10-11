import torch


def create_dataset(
    num_samples=1000,
    time_steps=20,
    input_size=10
):

    X = torch.zeros(
        time_steps,
        num_samples,
        input_size
    )

    y = torch.zeros(
        num_samples,
        dtype=torch.long
    )


    for i in range(num_samples):

        # Randomly select class
        label = torch.randint(
            0,
            2,
            (1,)
        ).item()

        y[i] = label


        if label == 0:

            # Early spikes
            start = 2
            end = 6

        else:

            # Late spikes
            start = 14
            end = 18


        # Add spikes
        X[
            start:end,
            i,
            :
        ] = torch.bernoulli(
            torch.full(
                (end - start, input_size),
                0.5
            )
        )


    return X, y
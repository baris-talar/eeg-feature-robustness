from moabb.datasets import PhysionetMI, BNCI2014_001
import matplotlib.pyplot as plt

def sanity_check():
    physionet = PhysionetMI()
    bci2a = BNCI2014_001()

    print("\n===== PHYSIONET =====")
    phys_sessions = physionet.get_data(subjects=[1])
    phys_raw = list(list(phys_sessions[1].values())[0].values())[0]

    phys_data = phys_raw.get_data()

    print("Shape:", phys_data.shape)
    print("Sampling rate:", phys_raw.info["sfreq"])
    print("Labels:", set(phys_raw.annotations.description))
    print("First 5 values:", phys_data[0][:5])

    # plot first channel
    plt.plot(phys_data[0][:1000])
    plt.title("PhysioNet - Channel 0 (first 1000 samples)")
    plt.show()


    print("\n===== BCI2A =====")
    bci_sessions = bci2a.get_data(subjects=[1])
    bci_raw = list(list(bci_sessions[1].values())[0].values())[0]

    bci_data = bci_raw.get_data()

    print("Shape:", bci_data.shape)
    print("Sampling rate:", bci_raw.info["sfreq"])
    print("Labels:", set(bci_raw.annotations.description))
    print("First 5 values:", bci_data[0][:5])

    plt.plot(bci_data[0][:1000])
    plt.title("BCI2a - Channel 0 (first 1000 samples)")
    plt.show()


if __name__ == "__main__":
    sanity_check()
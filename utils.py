import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime


# convert datetime string to integer (seconds since unix epoch)
def dt2int(dtstr, fmt="%Y-%m-%d %H:%M:%S"):
    return int(datetime.strptime(dtstr, fmt).strftime("%s"))


def visualize_meal(meal_time, meal_data, dt_min=None, dt_max=None, fmt="-"):
    plt.plot(meal_time, meal_data, fmt, color="orange")
    plt.xlabel("Epoch Time")
    plt.ylabel("Meal Size (g)")
    # plt.xlim(dt_min, dt_max)
    # plt.xticks(np.arange(dt_min, dt_max, 10000), rotation=90)


# visualize insulin data
def visualize_insulin(insulin_time, insulin_data, dt_min=None, dt_max=None, fmt="-"):
    plt.plot(insulin_time, insulin_data, fmt, color="green")
    plt.xlabel("Epoch Time")
    plt.ylabel("Insulin Delivered (mU)")
    # plt.xlim(dt_min, dt_max)
    # plt.xticks(np.arange(dt_min, dt_max, 10000), rotation=90)


# visualize cgm data
def visualize_cgm(cgm_time, cgm_data, dt_min=None, dt_max=None, fmt="-"):
    plt.plot(cgm_time, cgm_data, fmt, color="red")
    plt.xlabel("Epoch Time")
    plt.ylabel("BGL (mg/dL)")
    # plt.xlim(dt_min, dt_max)
    # plt.xticks(np.arange(dt_min, dt_max, 10000), rotation=90)


# visualize all data
def visualize(cgm_data, meal_data, insulin_data, dt_min=None, dt_max=None, fmt="-"):
    plt.figure(figsize=(8, 10))

    plt.subplot(311)
    visualize_meal(*meal_data, dt_min, dt_max, fmt)

    plt.subplot(312)
    visualize_insulin(*insulin_data, dt_min, dt_max, fmt)

    plt.subplot(313)
    visualize_cgm(*cgm_data, dt_min, dt_max, fmt)

    plt.tight_layout()
    plt.show()

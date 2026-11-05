import os
import time
from scripts.emitter_position_optimizer import BacklashSafeEmitterOptimizer


TCSPC_DEVICE = HydraHarp
StepperDevice = stepper111
SpectroDevice = stage3
CHANNEL = 2   


X_AXIS = 0
Y_AXIS = 1


HALF_RANGE = {
    X_AXIS: 0.0015,
    Y_AXIS: 0.0015,
}

STEP_SIZE = {
    X_AXIS: 0.0005,
    Y_AXIS: 0.0005,
}

BACKLASH_OVERSHOOT = {
    X_AXIS: 0.0005,
    Y_AXIS: 0.0005,
}

APPROACH_DIRECTION = {
    X_AXIS: +1,
    Y_AXIS: +1,
}

DWELL_TIME = 0.25

# First run can be False if you only want to scan/check.
# Set True to actually move to the optimized position.
MOVE_TO_RESULT = True

OUT_DIR = "" #Output path
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Hardware adapter functions
# ============================================================

def move_absolute(axis, position):
    StepperDevice.move_absolute(axis, position)


def get_position(axis):
    return StepperDevice.get_all()[0][axis]


def get_count_rate():
    return TCSPC_DEVICE.get_channels_rate()[CHANNEL - 1]


def move_xy_safe(x, y):
    move_absolute(X_AXIS, x)
    time.sleep(0.1)
    move_absolute(Y_AXIS, y)
    time.sleep(0.1)

def run_optimizer_with_retries(
    optimizer,
    max_retries=2,
    retry_wait=5.0,
):
    initial_x = get_position(X_AXIS)
    initial_y = get_position(Y_AXIS)

    best_result = None

    for attempt in range(max_retries + 1):
        print(f"\nOptimizer attempt {attempt + 1}/{max_retries + 1}")

        result = optimizer.scan_and_optimize(
            half_range=HALF_RANGE,
            step_size=STEP_SIZE,
            dwell_time=DWELL_TIME,

            localization_method="centroid",
            centroid_fraction=0.5,

            move_to_result=MOVE_TO_RESULT,

            reject_edge_maximum=True,
            min_peak_over_background=1.05,
            min_peak_rate=0.0,

            max_allowed_shift={
                X_AXIS: HALF_RANGE[X_AXIS],
                Y_AXIS: HALF_RANGE[Y_AXIS],
            },

            verify_final_position=True,
            final_verify_dwell_time=DWELL_TIME,
            
            save_hdf5_path=hdf5_path,
            hdf5_group_name="optimizer_test",

            plot_result=True,
            show_plot=True,
            plot_block=True,
            save_plot=False,
        )

        best_result = result

        if result.accepted:
            print("Optimization accepted.")
            return result

        print("Optimization rejected.")
        print(f"Reason: {result.reason}")

        print("Returning to initial position.")
        move_xy_safe(initial_x, initial_y)

        if attempt < max_retries:
            print(f"Waiting {retry_wait} s before retry...")
            time.sleep(retry_wait)

    print("\nAll optimizer attempts failed.")
    print("Emitter may be dark, blinking, or lost.")
    print("Stage left at initial position.")

    return best_result



try:
    TCSPC_DEVICE.stop_measurement()
except Exception:
    pass

try:
    SpectroDevice.set_exit_port("side")
except Exception:
    pass

print("\nStarting optimizer test")
print("--------------------------------")
print(f"Initial X: {get_position(X_AXIS):.6f}")
print(f"Initial Y: {get_position(Y_AXIS):.6f}")
print(f"Initial counts: {get_count_rate():.0f} cps")
print("--------------------------------")

optimizer = BacklashSafeEmitterOptimizer(
    move_absolute=move_absolute,
    get_position=get_position,
    get_count_rate=get_count_rate,
    x_axis=X_AXIS,
    y_axis=Y_AXIS,
    approach_direction=APPROACH_DIRECTION,
    backlash_overshoot=BACKLASH_OVERSHOOT,
    settling_time=0.1,
    sample_interval=0.05,
    verbose=True,
)

timestamp = time.strftime("%Y%m%d_%H%M%S")

hdf5_path = os.path.join(OUT_DIR, f"optimizer_test_{timestamp}.hdf5")
plot_path = os.path.join(OUT_DIR, f"optimizer_test_{timestamp}.png")

# result = optimizer.scan_and_optimize(
#     half_range=HALF_RANGE,
#     step_size=STEP_SIZE,
#     dwell_time=DWELL_TIME,

#     localization_method="centroid",
#     centroid_fraction=0.5,

#     move_to_result=MOVE_TO_RESULT,

#     reject_edge_maximum=True,
#     min_peak_over_background=1.02,
#     min_peak_rate=0.0,

#     max_allowed_shift={
#         X_AXIS: HALF_RANGE[X_AXIS],
#         Y_AXIS: HALF_RANGE[Y_AXIS],
#     },

#     verify_final_position=True,
#     final_verify_dwell_time=DWELL_TIME,

#     save_hdf5_path=hdf5_path,
#     hdf5_group_name="optimizer_test",

#     plot_result=True,
#     show_plot=True,
#     plot_block=True,
#     save_plot=True,
#     save_plot_path=plot_path,
#     plot_title="Real optimizer test",
# )

result = run_optimizer_with_retries(
    optimizer,
    max_retries=2,
    retry_wait=5.0,
)

print("\nOptimizer finished")
print("--------------------------------")
print(f"Accepted: {result.accepted}")
print(f"Reason: {result.reason}")
print(f"Old position      : {result.old_position}")
print(f"Estimated position: {result.estimated_position}")
print(f"Final position    : {result.new_position}")
print(f"Initial rate: {result.initial_rate:.0f} cps")
print(f"Final rate  : {result.final_rate:.0f} cps")
print(f"Peak rate   : {result.peak_rate:.0f} cps")
print("--------------------------------")
print(f"Saved HDF5: {hdf5_path}")
print(f"Saved plot: {plot_path}")
print("--------------------------------\n")
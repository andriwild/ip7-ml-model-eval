import subprocess

temp_result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
clock_result = subprocess.run(['vcgencmd', 'measure_clock', 'arm'], capture_output=True, text=True)

print("Temp: ", temp_result.stdout)
print("Clock: ", clock_result.stdout)

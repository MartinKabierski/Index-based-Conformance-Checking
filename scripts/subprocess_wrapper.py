import subprocess
import re
from tqdm import tqdm


# Executes the programs by Florian Marx as python subprocesses and returns the runtimes

def xes_to_txt(input="Road_Traffic_Fines_Management_Process.xes", output="Road_Traffic_Fines_Management_Process.txt"):
    print("Converting " + input + " to " + output)

    # The program (“./xes_to_txt”) is executed.
    # capture_output=True ensures that stdout and stderr are intercepted.
    # text=True converts the output into strings.
    result = subprocess.run(["./xes_to_txt",
                             "-i", input,
                             "-o", output,
                             "--version-check", "false"], capture_output=True, text=True)

    stdout_output = result.stdout
    stderr_output = result.stderr

    print("Standardausgabe:", stdout_output)
    if stderr_output != "":
        print("Fehlerausgabe:", stderr_output)
    print("-----------------------------------------------")

    # Extract the runtime
    numbers = re.findall(r"(\d+(?:\.\d+)?)+\[ms\]", stdout_output)
    conversion_time = float(numbers[-1])

    print(".xes to .txt conversion:", conversion_time, "ms")
    print("-----------------------------------------------")

    return conversion_time


def run_alignment_ibf(params=None, search="../output/Road_Traffic_Fines_Management_Process.txt", traces="../output/traces_Road_Traffic_Fines_Management_Process_noise0.2_max_loop1_trace_length22.txt",
                      output="test.csv", shortest_path=0, log_length=0):

    if params is None:
        params = {}

    print("Aligning " + search + " with " + traces)

    # The program (“./alignment_txt”) is executed.
    # capture_output=True ensures that stdout and stderr are intercepted.
    # text=True converts the output into strings.
    proc = subprocess.Popen(["./alignment_txt",
                             "-s", search,
                             "-t", traces,
                             "-o", output,
                             "-k", str(params["k"]),
                             "--num_hashes", str(params["num_hashes"]),
                             "--number_buckets_calc", str(params["buckets_calc"]),
                             "--bucketing", str(params["bucketing"]),
                             "--omh_w", str(params["omh_w"]),
                             "--omh_seed", str(params["omh_seed"]),
                             "--sorting", str(params["sorting"]),
                             "--shortest_path", str(shortest_path),
                             "--bucket_limit", str(params["bucket_limit"]),
                             #"-v",
                             "--version-check", "false"],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True,
                              bufsize=1,
                              )

    stdout_lines=[]

    # only lines with int values
    pattern = re.compile(r"^\d+$")

    # Process bar for the alignment
    with tqdm(total=log_length, unit=" lines", desc="Processing") as pbar:
        for line in proc.stdout:
            stdout_lines.append(line)
            if pattern.match(line.strip()):
                pbar.update(1)

    proc.wait()

    # stderr if error occurs
    stderr = proc.stderr.read()

    stdout_output = "".join(stdout_lines)
    stderr_output = stderr

    print(stderr_output)

    # Extract the runtime
    numbers = re.findall(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\[ms\]", stdout_output)

    generation_time = float(numbers[0])
    search_time = float(numbers[1])

    print("Bloom filter creation:", generation_time, "ms")
    print("IBF Search:", search_time, "ms")
    print("-----------------------------------------------")

    return generation_time, search_time


if __name__ == '__main__':
    run_alignment_ibf()
import logging
import time


def progress_iterations(fn, repetitions, quiet=False, **args):
    results = []
    if not quiet:
        print(f'Execute function {repetitions} times ...')
    start_time = time.time()
    for i in range(0, repetitions):
        try:
            result = fn(**args)
        except Exception as e:
            logging.error(e)
            break
        #        if i % (repetitions // 100) == 0:
        if not quiet:
            print(f'\r Progress: {i / repetitions * 100:3.0f}%', end="")
        if result is not None:
            results.append(result)
        else:
            break

    if not quiet:
        print('\r Progress: 100%')
        print(f'Executed function {repetitions} times in {(time.time() - start_time):.2f} seconds')
    return results

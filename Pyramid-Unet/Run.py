import subprocess
import os
import time
import logging
import sys
import shutil
from pathlib import Path


def setup_logger():
    """Configure the logger"""
    logger = logging.getLogger('file_processor')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Output to console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Output to file
    log_file = 'file_processor.log'
    fh = logging.FileHandler(log_file, encoding='utf-8')  # Specify UTF-8 encoding for log files
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Log file: {os.path.abspath(log_file)}")
    return logger


def is_file_ready(file_path):
    """Check if the file is accessible and complete"""
    if not os.path.exists(file_path):
        return False

    try:
        # Check if the file is readable
        with open(file_path, 'rb') as f:
            # Try to read the end of the file to check completeness
            f.seek(-1024, 2) if os.path.getsize(file_path) > 1024 else f.seek(0)
            f.read(1024)
        return True
    except Exception as e:
        logging.getLogger('file_processor').warning(f"File check failed: {str(e)}")
        return False


def wait_for_file_path(file_path, check_interval=1, timeout=3600):
    """
    Check if the file path exists and is accessible. If not, wait until the file path exists.
    Also displays the running time of the program in real-time.

    :param file_path: The file path to check
    :param check_interval: The time interval (in seconds) for checking the file path, default is 1 second
    :param timeout: The maximum waiting time (in seconds), default is 1 hour
    :return: True if the file exists and is accessible, False if timed out
    """
    start_time = time.time()
    logger = logging.getLogger('file_processor')

    logger.info(f"Waiting for file: {file_path}")

    while not os.path.exists(file_path):
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logger.error(f"Timeout: File path '{file_path}' did not appear within {timeout} seconds")
            return False

        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(f"\rProgram running time: {formatted_time} - File path '{file_path}' does not exist, waiting for file...", end='', flush=True)
        time.sleep(check_interval)

    print(f"\nFile '{file_path}' found, verifying file integrity...")

    # After the file exists, wait for the file to be accessible (not locked by other programs)
    max_attempts = 30
    for attempt in range(max_attempts):
        if is_file_ready(file_path):
            logger.info(f"File verification passed: {file_path}")
            return True

        elapsed_time = time.time() - start_time
        logger.warning(f"File temporarily inaccessible (Attempt {attempt + 1}/{max_attempts}, waited {int(elapsed_time)} seconds)")
        time.sleep(check_interval)

    logger.error(f"File '{file_path}' exists but cannot be accessed, may be locked or incomplete")
    return False


def copy_to_temp(input_file):
    """Copy the file to a temporary location to avoid modifications to the source file during processing"""
    logger = logging.getLogger('file_processor')

    # Create a temporary directory
    temp_dir = os.path.join(os.environ.get('TEMP', '/tmp'), 'file_processor')
    os.makedirs(temp_dir, exist_ok=True)

    # Generate a unique temporary file name
    file_name = os.path.basename(input_file)
    temp_file = os.path.join(temp_dir, f"{int(time.time())}_{file_name}")

    try:
        logger.info(f"Copying file to temporary location: {temp_file}")
        shutil.copy2(input_file, temp_file)  # Use copy2 to preserve file metadata
        return temp_file
    except Exception as e:
        logger.error(f"Failed to copy file: {str(e)}")
        return None


def process_file(input_file, output_file=None):
    """Process a single file"""
    logger = logging.getLogger('file_processor')

    # Wait for the input file
    if not wait_for_file_path(input_file):
        return False

    # Copy the file to a temporary location for processing
    temp_file = copy_to_temp(input_file)
    if not temp_file:
        logger.error(f"Unable to create temporary file, skipping processing: {input_file}")
        return False

    try:
        # Use a normalized file path
        normalized_path = Path(temp_file).as_posix()
        logger.info(f"Using normalized path: {normalized_path}")

        # Execute the prediction script, passing the normalized path
        result = subprocess.run(
            ['python', 'predict.py', normalized_path],  # Pass the file path as a parameter
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}  # Set Python encoding environment variable
        )

        logger.info(f"Prediction script executed successfully: {result.stdout}")
        print("The file writing has been completed.")

        # Check if the output file is generated (if required)
        if output_file and not os.path.exists(output_file):
            logger.warning(f"Expected output file '{output_file}' was not generated")

    except subprocess.CalledProcessError as e:
        logger.error(f"Prediction script execution failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"An unknown error occurred while executing the prediction script: {str(e)}", exc_info=True)
        return False
    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"Temporary file cleaned up: {temp_file}")
            except Exception as e:
                logger.warning(f"Unable to clean up temporary file: {str(e)}")

    # Delete the original input file (if processing was successful)
    try:
        os.remove(input_file)
        logger.info(f"Input file deleted: {input_file}")
        print("The file has been removed.")
    except Exception as e:
        logger.error(f"Unable to delete input file: {str(e)}")
        return False

    return True


def main():
    """Main program entry point"""
    # Configure logging
    setup_logger()
    logger = logging.getLogger('file_processor')

    # Print system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"System encoding: {sys.getdefaultencoding()}")
    logger.info(f"File system encoding: {sys.getfilesystemencoding()}")

    logger.info("==== File processing program started ====")

    # Configure file paths
    input_file = r"C:\Users\17105\Desktop\data-transmit\1.tif.tif"
    output_file = r"C:\Users\17105\Desktop\data-transmit\output_table.csv"

    # Ensure directories exist
    for path in [input_file, output_file]:
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Directory created: {dir_path}")

    # Main loop
    try:
        while True:
            logger.info("Waiting for new file...")
            success = process_file(input_file, output_file)

            # If processing fails, wait for a while and retry
            if not success:
                logger.warning("File processing failed, waiting 10 seconds before retrying")
                time.sleep(10)

            # Wait for the next file
            logger.info("Ready to process next file...")
            time.sleep(1)  # Avoid high CPU usage

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.critical(f"A fatal error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
def handle_corrupt_file(exception, video_path, log_path):
    # TODO clean this code/make it more normal
    with open(log_path, 'a') as f:
        f.write(f"\n{video_path}")

    error_text = f"""
    Error on video: {video_path}
    The video path has been logged to {log_path} and will be ignored in future runs.
    Exception string: {str(exception)}
    """

    raise Exception(error_text)
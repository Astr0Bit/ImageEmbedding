
import os

# Image processing and plotting libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Google MediaPipe imports
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageEmbedder, ImageEmbedderOptions

# Validation
from validation import Validation as Val

class ImageEmbedding:

    """
    Class for performing image embedding using Google's MediaPipe library.
    """

    root = os.path.dirname(os.path.abspath(__file__))
    target_width = 700
    target_height = 600

    def __init__(self, l2_normalize: bool = True, 
                 quantize: bool = True) -> None:
        """
        Args: 
            l2_normalize: Whether to L2 normalize the embeddings.
            quantize: Whether to quantize the embeddings.

        Returns:
            None
        """
        self.model_path = os.path.join(
            ImageEmbedding.root, "mobilenet_v3_large_075_224_embedder.tflite")
        self.base_options = BaseOptions(model_asset_path=self.model_path)
        self.options = ImageEmbedderOptions(base_options=self.base_options,
                                            l2_normalize=l2_normalize,
                                            quantize=quantize)

    def predict_images(self, IMAGE_FILENAMES: list | tuple[str, str]) -> None:
        """
        Predicts the embeddings of the input images.

        Args:
            image_paths: A list of paths to the input images.

        Returns:
            None

        Raises:
            TypeError: If IMAGE_FILENAMES is not a list or tuple.
            ValueError: If the number of image file paths is not 2.
            FileNotFoundError: If an image file path does not exist.
        """

        # Check that IMAGE_FILENAMES is a list or tuple
        # and that it contains exactly 2 image file paths
        Val.val_iter(IMAGE_FILENAMES, "IMAGE_FILENAMES")
        
        # Check that both paths exist and are images
        for image in IMAGE_FILENAMES:
            Val.val_path(image)
            Val.val_filetype(image, (".png", ".jpg", ".jpeg"))
        
        target_width = ImageEmbedding.target_width
        target_height = ImageEmbedding.target_height

        # Create the image embedder
        with ImageEmbedder.create_from_options(self.options) as embedder:
    
            # format images
            first_image_mat = cv2.imread(IMAGE_FILENAMES[0])
            second_image_mat = cv2.imread(IMAGE_FILENAMES[1])
            first_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                                   data=first_image_mat)
            second_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                                    data=second_image_mat)

            # Resize the images
            first_resized = cv2.resize(first_image_mat, 
                                      (target_width, target_height))
            second_resized = cv2.resize(second_image_mat, 
                                      (target_width, target_height))

            first_embedding_result = embedder.embed(first_image)
            second_embedding_result = embedder.embed(second_image)

            # calculate similarity between images
            similarity = ImageEmbedder.cosine_similarity(
                first_embedding_result.embeddings[0],
                second_embedding_result.embeddings[0])
            
            # Create a side by side image
            side_by_side = cv2.hconcat([first_resized, second_resized])

            # Add similarity text to the frame
            frame_center = (target_width // 2 + 200, target_height - 50)

            # Display the concatenated frames with text using matplotlib
            plt.imshow(cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB))
            plt.text(frame_center[0], frame_center[1], 
                     f"Similarity: {similarity:.2f}", 
                     fontsize=12, color="white")
            
            plt.show()
    
    def predict_video(self, VIDEO_FILENAMES: list | tuple[str, str], 
                      save: bool = False, 
                      output_file_name: str = "similarity_side_by_side") -> None:
        """
        Method for taking in a list or tuple of 2 video file paths,
        and calculating the similarity between the frames of the two videos.

        Args:
            VIDEO_FILENAMES: A list or tuple of 2 video file paths.
            save: A boolean indicating whether to save the output video.
        
        Returns:
            None

        Raises:
            TypeError: If VIDEO_FILENAMES is not a list or tuple.
            ValueError: If the number of video file paths is not 2.
            FileNotFoundError: If a video file path does not exist.
        """

        # Check that VIDEO_FILENAME is a list or tuple
        # and that it contains exactly 2 video file paths
        Val.val_iter(VIDEO_FILENAMES, "VIDEO_FILENAMES")
        
        # Check that both paths exist and are videos
        for video in VIDEO_FILENAMES:
            Val.val_path(video)
            Val.val_filetype(video, (".mp4", ".avi"))
            
        # Prepare the plot for displaying the similarity for each frame
        fig = plt.figure()
        frame_number = 0
        frame_numbers_list = [0]
        similarities_list = [0]

        # Get number of frames for each video
        VIDEO_FRAMES = []
        side_by_side_frames = []

        for video in VIDEO_FILENAMES:
            cap = cv2.VideoCapture(video)

            FRAMES = []
            for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, frame = cap.read()
                FRAMES.append(frame)

            print(f"Frames found in {video}: {len(FRAMES)}")
            VIDEO_FRAMES.append(FRAMES)
            cap.release()

        # Handle missmatch in frames -> adjust number of frames
        MIN_FRAMES = min([len(frames) for frames in VIDEO_FRAMES])
        print("Adjusting frames for both videos. "\
              f"New number of frames: {MIN_FRAMES}")
        VIDEO_FRAMES = [frames[:MIN_FRAMES] for frames in VIDEO_FRAMES]

        target_width = ImageEmbedding.target_width
        target_height = ImageEmbedding.target_height

        with ImageEmbedder.create_from_options(self.options) as embedder:
            for i in range(MIN_FRAMES):
                vid1_frame_mat = VIDEO_FRAMES[0][i]
                vid2_frame_mat = VIDEO_FRAMES[1][i]

                # Resize the frames to the target dimensions
                vid1_frame_resized = cv2.resize(vid1_frame_mat, 
                                               (target_width, target_height))
                vid2_frame_resized = cv2.resize(vid2_frame_mat, 
                                               (target_width, target_height))
                
                # get embeddings
                vid1_frame = mp.Image(image_format=mp.ImageFormat.SRGB, 
                                      data=vid1_frame_mat)
                vid2_frame = mp.Image(image_format=mp.ImageFormat.SRGB, 
                                      data=vid2_frame_mat)
                first_embedding_result = embedder.embed(vid1_frame)
                second_embedding_result = embedder.embed(vid2_frame)

                # calculate similarity between frames
                similarity = ImageEmbedder.cosine_similarity(
                    first_embedding_result.embeddings[0],
                    second_embedding_result.embeddings[0])
                
                # Append similarity and current frame number to lists
                similarities_list.append(similarity)
                frame_numbers_list.append(i)

                # Plotting similarity data
                # print(frame_numbers_list, similarities_list)
                plt.plot(frame_numbers_list, similarities_list, color="blue")
                plt.title("Similarity between frames")
                plt.ylabel("Similarity")
                plt.xlabel("Frame number")
                fig.canvas.draw()

                # Convert matplotlib figure to OpenCV image
                plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (4,)) # RGBA format
                plot = cv2.cvtColor(plot, cv2.COLOR_RGBA2BGR) # Convert to BGR for OpenCV

                # Resize the plot to the target dimensions
                plot_resized = cv2.resize(plot, (target_width * 2, target_height))
                                
                # TODO: Display the frames side by side using OpenCV
                side_by_side = cv2.hconcat([vid1_frame_resized, 
                                            vid2_frame_resized])
                
                side_by_side_plot = cv2.vconcat([side_by_side, plot_resized])

                # Add similarity text to the frame
                frame_center = (target_width // 2 + 200, target_height - 50)
                cv2.putText(
                    side_by_side_plot,
                    f"Similarity: {similarity:.2f}",
                    frame_center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), 2)

                if save:
                    side_by_side_frames.append(side_by_side_plot)

                # Display the concatenated frames
                cv2.imshow("Frames", side_by_side_plot)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if save:
            writer = cv2.VideoWriter(
                f"{output_file_name}.mp4",
                cv2.VideoWriter_fourcc(*"VIDX"),
                30, (target_width*2, target_height*2))
            
            for frame in side_by_side_frames:
                writer.write(frame)
            writer.release()
        cv2.destroyAllWindows()
        plt.close()
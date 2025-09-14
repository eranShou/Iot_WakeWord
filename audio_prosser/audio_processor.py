#!/usr/bin/env python3
"""
Voice Sample Audio Processor

This program processes voice sample recordings from subfolders within a main directory.
It automatically detects word segments in WAV files and splits them into individual recordings
with continuous numbering across all files in each subfolder (1-5, 6-10, 11-15, etc.).

Author: AI Assistant
Date: September 2025
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Audio processing imports
try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    PYDUB_AVAILABLE = True
except ImportError:
    print("Warning: pydub not available. Please install with: pip install pydub")
    PYDUB_AVAILABLE = False

# Progress visualization imports
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: tqdm not available. Please install with: pip install tqdm")
    TQDM_AVAILABLE = False

# Configuration imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("Warning: pyyaml not available. Configuration file support disabled. ")
    YAML_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AudioProcessorError(Exception):
    """Custom exception for audio processing errors."""
    pass


class AudioProcessor:
    """
    Main class for processing voice sample audio files.

    Handles directory traversal, audio analysis, splitting, and file management.
    """

    def __init__(self, input_dir: str, output_dir: str = "ProcessedSamples"):
        """
        Initialize the AudioProcessor.

        Args:
            input_dir: Path to the main SoundSamples directory
            output_dir: Path to the output directory for processed files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.expected_segments = 5

        # Validate input directory
        if not self.input_dir.exists():
            raise AudioProcessorError(f"Input directory does not exist: {input_dir}")

        if not self.input_dir.is_dir():
            raise AudioProcessorError(f"Input path is not a directory: {input_dir}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"AudioProcessor initialized with input: {input_dir}, output: {output_dir}")

        # Load configuration
        self.config = self._load_config()

        # Apply configuration settings
        self._apply_config_settings()

    def _load_config(self) -> dict:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        default_config = {
            'processing': {'expected_segments': 5, 'output_directory': 'ProcessedSamples'},
            'silence_detection': {
                'min_silence_len': 500,
                'silence_thresh': -40,
                'seek_step': 10,
                'retry_attempts': [
                    {'min_silence_len': 500, 'silence_thresh': -40, 'seek_step': 10},
                    {'min_silence_len': 300, 'silence_thresh': -35, 'seek_step': 5},
                    {'min_silence_len': 700, 'silence_thresh': -45, 'seek_step': 15},
                    {'min_silence_len': 200, 'silence_thresh': -30, 'seek_step': 5},
                ]
            },
            'validation': {
                'min_duration': 1000,
                'silence_threshold': -50,
                'max_rms': 1000000,
                'sample_rate': 100
            },
            'segment_filtering': {
                'min_segment_length': 200,
                'max_segment_length': 3000,
                'padding': 100
            }
        }

        if not YAML_AVAILABLE:
            logger.info("YAML not available, using default configuration")
            return default_config

        config_path = Path("config.yaml")
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_path}")

                # Merge user config with defaults
                merged_config = self._merge_configs(default_config, user_config)
                return merged_config

            except Exception as e:
                logger.warning(f"Error loading configuration: {e}. Using defaults.")
                return default_config
        else:
            logger.info("No config.yaml found, using default configuration")
            return default_config

    def _merge_configs(self, default: dict, user: dict) -> dict:
        """
        Recursively merge user configuration with defaults.

        Args:
            default: Default configuration
            user: User configuration

        Returns:
            Merged configuration
        """
        if user is None:
            return default

        result = default.copy()

        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_config_settings(self):
        """
        Apply configuration settings to instance variables.
        """
        # Update instance variables from config
        if 'processing' in self.config:
            if 'expected_segments' in self.config['processing']:
                self.expected_segments = self.config['processing']['expected_segments']

        # Update output directory if specified in config
        if 'processing' in self.config and 'output_directory' in self.config['processing']:
            config_output = Path(self.config['processing']['output_directory'])
            if config_output != Path("ProcessedSamples"):  # Only change if different from default
                self.output_dir = config_output
                self.output_dir.mkdir(parents=True, exist_ok=True)

    def discover_subfolders(self) -> List[Path]:
        """
        Discover all subfolders within the input directory.

        Returns:
            List of Path objects for subfolders containing WAV files
        """
        subfolders = []

        try:
            for item in self.input_dir.iterdir():
                if item.is_dir():
                    # Check if the subfolder contains any WAV files
                    wav_files = list(item.glob("*.wav"))
                    if wav_files:
                        subfolders.append(item)
                        logger.debug(f"Found subfolder with WAV files: {item.name} ({len(wav_files)} files)")
                    else:
                        logger.debug(f"Skipping subfolder without WAV files: {item.name}")

            logger.info(f"Discovered {len(subfolders)} subfolders with WAV files")
            return subfolders

        except PermissionError as e:
            raise AudioProcessorError(f"Permission denied accessing directory: {e}")
        except Exception as e:
            raise AudioProcessorError(f"Error discovering subfolders: {e}")

    def discover_audio_files(self, subfolder: Path) -> List[Path]:
        """
        Discover all WAV audio files in a specific subfolder.

        Args:
            subfolder: Path to the subfolder to scan

        Returns:
            List of Path objects for WAV files in the subfolder
        """
        audio_files = []

        try:
            for wav_file in subfolder.glob("*.wav"):
                if wav_file.is_file():
                    audio_files.append(wav_file)
                    logger.debug(f"Found audio file: {wav_file.name}")

            logger.info(f"Found {len(audio_files)} audio files in {subfolder.name}")
            return audio_files

        except PermissionError as e:
            raise AudioProcessorError(f"Permission denied accessing files in {subfolder.name}: {e}")
        except Exception as e:
            raise AudioProcessorError(f"Error discovering audio files in {subfolder.name}: {e}")

    def analyze_audio_file(self, audio_file: Path, retry_params: List[dict] = None) -> Tuple[List[Tuple[int, int]], AudioSegment]:
        """
        Analyze an audio file to detect word segments based on silence detection.
        Includes retry mechanism with adaptive parameters.

        Args:
            audio_file: Path to the WAV audio file
            retry_params: List of parameter dictionaries for retry attempts

        Returns:
            Tuple of (list of segment time ranges, original audio segment)

        Raises:
            AudioProcessorError: If audio file cannot be loaded or analyzed
        """
        if not PYDUB_AVAILABLE:
            raise AudioProcessorError("pydub library is required for audio processing")

        try:
            # Load the audio file
            audio = AudioSegment.from_wav(audio_file)
            logger.debug(f"Loaded audio file: {audio_file.name}, duration: {len(audio)}ms")

            # Validate audio file
            validation_result = self._validate_audio_file(audio, audio_file)
            if not validation_result['is_valid']:
                logger.warning(f"Audio validation failed for {audio_file.name}: {validation_result['reason']}")
                # Return empty segments list for invalid files
                return [], audio

            # Use retry parameters from configuration
            if retry_params is None:
                if 'silence_detection' in self.config and 'retry_attempts' in self.config['silence_detection']:
                    retry_params = self.config['silence_detection']['retry_attempts']
                else:
                    # Fallback to default parameters
                    retry_params = [
                        # Original parameters
                        {"min_silence_len": 500, "silence_thresh": -40, "seek_step": 10},
                        # More sensitive (shorter silences)
                        {"min_silence_len": 300, "silence_thresh": -35, "seek_step": 5},
                        # Less sensitive (longer silences)
                        {"min_silence_len": 700, "silence_thresh": -45, "seek_step": 15},
                        # Very sensitive for difficult cases
                        {"min_silence_len": 200, "silence_thresh": -30, "seek_step": 5},
                    ]

            best_segments = []
            best_score = 0

            # Try different parameter combinations
            for i, params in enumerate(retry_params):
                try:
                    logger.debug(f"Trying detection parameters (attempt {i+1}): {params}")

                    # Detect non-silent chunks with current parameters
                    nonsilent_chunks = detect_nonsilent(
                        audio,
                        min_silence_len=params["min_silence_len"],
                        silence_thresh=params["silence_thresh"],
                        seek_step=params["seek_step"]
                    )

                    logger.debug(f"Attempt {i+1}: Detected {len(nonsilent_chunks)} non-silent chunks")

                    # Filter and validate segments
                    valid_segments = self._filter_segments(nonsilent_chunks, audio)

                    # Score this attempt (prefer closer to expected segments)
                    if len(valid_segments) <= self.expected_segments:
                        score = len(valid_segments)  # Full score for exact or fewer segments
                    else:
                        # Penalty for too many segments
                        score = max(0, self.expected_segments - (len(valid_segments) - self.expected_segments))

                    logger.debug(f"Attempt {i+1}: {len(valid_segments)} valid segments (score: {score})")

                    # Keep track of best result
                    if score > best_score:
                        best_segments = valid_segments
                        best_score = score

                    # If we got exactly the expected number, stop trying
                    if len(valid_segments) == self.expected_segments:
                        logger.debug(f"Perfect match found on attempt {i+1}")
                        break

                except Exception as e:
                    logger.warning(f"Detection attempt {i+1} failed: {e}")
                    continue

            # Final validation and logging
            if len(best_segments) < self.expected_segments:
                logger.warning(
                    f"Best result: Expected {self.expected_segments} segments, "
                    f"detected {len(best_segments)} in {audio_file.name}"
                )
            elif len(best_segments) > self.expected_segments:
                logger.warning(
                    f"Best result: Detected {len(best_segments)} segments, "
                    f"using top {self.expected_segments} in {audio_file.name}"
                )
                # Use only the expected number of segments
                best_segments = best_segments[:self.expected_segments]
            else:
                logger.debug(f"Perfect: Detected exactly {self.expected_segments} segments in {audio_file.name}")

            return best_segments, audio

        except Exception as e:
            raise AudioProcessorError(f"Error analyzing audio file {audio_file.name}: {e}")

    def _filter_segments(self, chunks: List[Tuple[int, int]], audio: AudioSegment) -> List[Tuple[int, int]]:
        """
        Filter and validate detected audio segments.

        Args:
            chunks: List of (start, end) tuples in milliseconds
            audio: Original audio segment

        Returns:
            Filtered list of valid segments
        """
        valid_segments = []

        # Get filtering parameters from config
        min_segment_length = self.config.get('segment_filtering', {}).get('min_segment_length', 200)
        max_segment_length = self.config.get('segment_filtering', {}).get('max_segment_length', 3000)
        padding = self.config.get('segment_filtering', {}).get('padding', 100)

        for start, end in chunks:
            segment_length = end - start

            # Filter out segments that are too short or too long
            if min_segment_length <= segment_length <= max_segment_length:
                # Add padding around the segment to capture the full word
                padded_start = max(0, start - padding)
                padded_end = min(len(audio), end + padding)

                valid_segments.append((padded_start, padded_end))
                logger.debug(f"Valid segment: {padded_start}ms - {padded_end}ms ({segment_length}ms)")

        # If we have more than expected segments, try to select the best ones
        if len(valid_segments) > self.expected_segments:
            valid_segments = self._select_best_segments(valid_segments)

        return valid_segments

    def _select_best_segments(self, segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Select the best segments when more than expected are detected.

        Args:
            segments: List of (start, end) tuples

        Returns:
            Selected segments (up to expected_segments count)
        """
        # Sort by segment length (prefer medium-length segments)
        sorted_segments = sorted(segments, key=lambda x: abs((x[1] - x[0]) - 1000))  # Prefer ~1s segments

        # Return the top segments
        return sorted_segments[:self.expected_segments]

    def _validate_audio_file(self, audio: AudioSegment, audio_file: Path) -> dict:
        """
        Validate audio file for basic quality checks.

        Args:
            audio: Audio segment to validate
            audio_file: Path to the audio file

        Returns:
            Dictionary with validation results
        """
        validation = {'is_valid': True, 'reason': ''}

        # Get validation parameters from config
        min_duration = self.config.get('validation', {}).get('min_duration', 1000)
        silence_threshold = self.config.get('validation', {}).get('silence_threshold', -50)
        max_rms = self.config.get('validation', {}).get('max_rms', 1000000)
        sample_rate = self.config.get('validation', {}).get('sample_rate', 100)

        # Check minimum duration
        if len(audio) < min_duration:
            validation['is_valid'] = False
            validation['reason'] = f"Audio too short: {len(audio)}ms (minimum: {min_duration}ms)"
            return validation

        # Check for completely silent audio
        # Sample the audio at regular intervals to check for any sound
        has_audio = False

        for i in range(0, len(audio), sample_rate):
            if i >= len(audio):
                break
            sample = audio[i:i+100]  # 100ms sample
            if sample.dBFS > silence_threshold:
                has_audio = True
                break

        if not has_audio:
            validation['is_valid'] = False
            validation['reason'] = "Audio appears to be completely silent or corrupted"
            return validation

        # Check for corrupted audio (extreme values)
        try:
            # Get RMS (Root Mean Square) as a basic corruption check
            rms = audio.rms
            if rms == 0 or rms > max_rms:  # Unrealistic RMS values
                validation['is_valid'] = False
                validation['reason'] = f"Audio corruption detected (RMS: {rms})"
                return validation
        except Exception:
            validation['is_valid'] = False
            validation['reason'] = "Audio file appears corrupted (cannot calculate RMS)"
            return validation

        return validation

    def split_audio_file(self, audio_file: Path, segments: List[Tuple[int, int]],
                        audio: AudioSegment, subfolder_name: str, start_segment_num: int) -> List[Path]:
        """
        Split an audio file into segments and save them to the output directory.

        Args:
            audio_file: Original audio file path
            segments: List of (start, end) time ranges in milliseconds
            audio: Original audio segment
            subfolder_name: Name of the subfolder (used for naming)
            start_segment_num: Starting segment number for this file

        Returns:
            List of paths to the saved segment files

        Raises:
            AudioProcessorError: If splitting or saving fails
        """
        saved_files = []

        try:
            # Create subfolder in output directory
            output_subfolder = self.output_dir / subfolder_name
            output_subfolder.mkdir(parents=True, exist_ok=True)

            logger.info(f"Splitting {audio_file.name} into {len(segments)} segments")

            for i, (start, end) in enumerate(segments):
                # Calculate the actual segment number
                segment_num = start_segment_num + i

                # Extract the segment
                segment = audio[start:end]

                # Generate filename based on subfolder name and segment number
                filename = f"{subfolder_name}_{segment_num}.wav"
                output_path = output_subfolder / filename

                # Export the segment
                segment.export(output_path, format="wav")

                saved_files.append(output_path)
                logger.debug(f"Saved segment {segment_num}: {filename}")

            logger.info(f"Successfully split {audio_file.name} into {len(saved_files)} segments")
            return saved_files

        except Exception as e:
            raise AudioProcessorError(f"Error splitting audio file {audio_file.name}: {e}")

    def process_audio_file(self, audio_file: Path, subfolder_name: str, start_segment_num: int) -> Optional[List[Path]]:
        """
        Process a single audio file: analyze, split, and save segments.

        Args:
            audio_file: Path to the audio file to process
            subfolder_name: Name of the parent subfolder
            start_segment_num: Starting segment number for this file

        Returns:
            List of paths to saved segment files, or None if processing failed
        """
        try:
            logger.info(f"Processing audio file: {audio_file.name}")

            # Analyze the audio file to detect segments
            segments, audio = self.analyze_audio_file(audio_file)

            # Check if we have the expected number of segments
            if len(segments) < self.expected_segments:
                logger.warning(
                    f"Insufficient segments detected in {audio_file.name}: "
                    f"expected {self.expected_segments}, got {len(segments)}. Skipping file."
                )
                return None

            # If we have more segments, we already filtered them in analyze_audio_file
            if len(segments) > self.expected_segments:
                logger.warning(
                    f"Too many segments detected in {audio_file.name}: "
                    f"expected {self.expected_segments}, got {len(segments)}. Using best {self.expected_segments}."
                )

            # Split the audio file and save segments
            saved_files = self.split_audio_file(audio_file, segments, audio, subfolder_name, start_segment_num)

            logger.info(f"Successfully processed {audio_file.name}: created {len(saved_files)} segments")
            return saved_files

        except AudioProcessorError as e:
            logger.error(f"Failed to process {audio_file.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {audio_file.name}: {e}")
            return None

    def process_all_files(self) -> dict:
        """
        Process all audio files in all subfolders.

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total_subfolders': 0,
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_segments': 0
        }

        try:
            # Discover all subfolders
            subfolders = self.discover_subfolders()
            stats['total_subfolders'] = len(subfolders)

            logger.info(f"Starting processing of {len(subfolders)} subfolders")

            # Progress bar for subfolders
            subfolder_pbar = tqdm(
                subfolders,
                desc="Processing subfolders",
                unit="folder",
                disable=not TQDM_AVAILABLE
            )

            for subfolder in subfolder_pbar:
                subfolder_pbar.set_description(f"Processing {subfolder.name}")

                # Discover audio files in this subfolder
                audio_files = self.discover_audio_files(subfolder)
                stats['total_files'] += len(audio_files)

                subfolder_processed = 0
                subfolder_failed = 0

                # Track segment numbering for this subfolder
                current_segment_num = 1

                # Progress bar for files in this subfolder
                if TQDM_AVAILABLE:
                    file_pbar = tqdm(
                        audio_files,
                        desc=f"Files in {subfolder.name}",
                        unit="file",
                        leave=False
                    )

                    # Process each audio file
                    for audio_file in file_pbar:
                        result = self.process_audio_file(audio_file, subfolder.name, current_segment_num)

                        if result is not None:
                            stats['processed_files'] += 1
                            stats['total_segments'] += len(result)
                            subfolder_processed += 1

                            # Update the segment counter for the next file
                            current_segment_num += self.expected_segments
                        else:
                            stats['failed_files'] += 1
                            subfolder_failed += 1

                        # Update file progress bar
                        file_pbar.set_postfix({
                            'processed': subfolder_processed,
                            'failed': subfolder_failed
                        })

                    file_pbar.close()
                else:
                    # Process without progress bar if tqdm not available
                    for audio_file in audio_files:
                        result = self.process_audio_file(audio_file, subfolder.name, current_segment_num)

                        if result is not None:
                            stats['processed_files'] += 1
                            stats['total_segments'] += len(result)
                            subfolder_processed += 1

                            # Update the segment counter for the next file
                            current_segment_num += self.expected_segments
                        else:
                            stats['failed_files'] += 1
                            subfolder_failed += 1

                logger.info(
                    f"Subfolder {subfolder.name}: {subfolder_processed} processed, "
                    f"{subfolder_failed} failed"
                )

            # Final summary
            logger.info("=" * 50)
            logger.info("PROCESSING COMPLETE")
            logger.info(f"Subfolders processed: {stats['total_subfolders']}")
            logger.info(f"Total audio files: {stats['total_files']}")
            logger.info(f"Successfully processed: {stats['processed_files']}")
            logger.info(f"Failed files: {stats['failed_files']}")
            logger.info(f"Total segments created: {stats['total_segments']}")
            logger.info("=" * 50)

            return stats

        except AudioProcessorError as e:
            logger.error(f"Processing failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during processing: {e}")
            raise AudioProcessorError(f"Processing failed: {e}")

    def generate_processing_report(self, stats: dict, failed_files: List[dict] = None) -> str:
        """
        Generate a comprehensive processing report.

        Args:
            stats: Processing statistics dictionary
            failed_files: List of failed file information

        Returns:
            Formatted report string
        """
        from datetime import datetime

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("AUDIO PROCESSING REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Overall Statistics
        report_lines.append("OVERALL STATISTICS:")
        report_lines.append("-" * 30)
        report_lines.append(f"Total subfolders processed: {stats['total_subfolders']}")
        report_lines.append(f"Total audio files found:    {stats['total_files']}")
        report_lines.append(f"Files successfully processed: {stats['processed_files']}")
        report_lines.append(f"Files failed:              {stats['failed_files']}")
        report_lines.append(f"Total segments created:    {stats['total_segments']}")
        report_lines.append("")

        # Success Rate
        if stats['total_files'] > 0:
            success_rate = (stats['processed_files'] / stats['total_files']) * 100
            report_lines.append(f"Success Rate: {success_rate:.1f}%")
            report_lines.append("")

        # Performance Metrics
        if stats['processed_files'] > 0:
            avg_segments_per_file = stats['total_segments'] / stats['processed_files']
            report_lines.append("PERFORMANCE METRICS:")
            report_lines.append("-" * 30)
            report_lines.append(f"Average segments per file: {avg_segments_per_file:.1f}")
            report_lines.append(f"Expected segments per file: {self.expected_segments}")
            report_lines.append("")

        # Failure Analysis
        if failed_files and len(failed_files) > 0:
            report_lines.append("FAILURE ANALYSIS:")
            report_lines.append("-" * 30)

            # Categorize failures
            failure_categories = {}
            for failure in failed_files:
                category = failure.get('category', 'Unknown')
                if category not in failure_categories:
                    failure_categories[category] = []
                failure_categories[category].append(failure)

            for category, files in failure_categories.items():
                report_lines.append(f"{category}: {len(files)} files")
                for failure in files[:5]:  # Show first 5 failures per category
                    report_lines.append(f"  - {failure['filename']}: {failure.get('reason', 'Unknown error')}")
                if len(files) > 5:
                    report_lines.append(f"  ... and {len(files) - 5} more files")
                report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 30)

        if stats['total_files'] > 0:
            success_rate = (stats['processed_files'] / stats['total_files']) * 100
            if success_rate < 75:
                report_lines.append("⚠️  LOW SUCCESS RATE - Consider:")
                report_lines.append("   • Checking audio file quality and format")
                report_lines.append("   • Adjusting silence detection parameters")
                report_lines.append("   • Verifying ffmpeg installation")
            elif success_rate < 90:
                report_lines.append("ℹ️  MODERATE SUCCESS RATE - Optimization opportunities:")
                report_lines.append("   • Fine-tune silence detection parameters")
                report_lines.append("   • Consider audio normalization")
            else:
                report_lines.append("✅ HIGH SUCCESS RATE - Great results!")

        report_lines.append("")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def save_report_to_file(self, report: str, output_dir: Path) -> Path:
        """
        Save the processing report to a file.

        Args:
            report: Report content
            output_dir: Directory to save the report

        Returns:
            Path to the saved report file
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"processing_report_{timestamp}.txt"
        report_path = output_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Processing report saved to: {report_path}")
        return report_path


def main():
    """Main entry point for the audio processor."""
    parser = argparse.ArgumentParser(
        description="Process voice sample recordings and split into individual word segments with continuous numbering (1-5, 6-10, 11-15, etc.)."
    )
    parser.add_argument(
        "input_dir",
        help="Path to the SoundSamples directory containing subfolders with WAV files"
    )
    parser.add_argument(
        "-o", "--output",
        default="ProcessedSamples",
        help="Output directory for processed files (default: ProcessedSamples)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Check if required libraries are available
        if not PYDUB_AVAILABLE:
            logger.error(
                "pydub library is required but not installed. "
                "Please install with: pip install pydub"
            )
            sys.exit(1)

        if not TQDM_AVAILABLE:
            logger.warning(
                "tqdm library not available. Progress bars will be disabled. "
                "Install with: pip install tqdm"
            )

        if not YAML_AVAILABLE:
            logger.warning(
                "pyyaml library not available. Configuration file support disabled. "
                "Install with: pip install pyyaml"
            )

        # Initialize the processor
        processor = AudioProcessor(args.input_dir, args.output)
        logger.info("Audio processor initialized successfully")

        # Run the processing pipeline
        stats = processor.process_all_files()

        # Generate comprehensive report
        report = processor.generate_processing_report(stats)

        # Save report to file
        report_path = processor.save_report_to_file(report, Path(args.output))

        # Print final summary to console
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Subfolders found:     {stats['total_subfolders']}")
        print(f"Audio files found:    {stats['total_files']}")
        print(f"Files processed:      {stats['processed_files']}")
        print(f"Files failed:         {stats['failed_files']}")
        print(f"Segments created:     {stats['total_segments']}")
        print(f"Report saved to:      {report_path}")
        print("=" * 50)

        # Print success/failure message
        if stats['processed_files'] > 0:
            success_rate = (stats['processed_files'] / stats['total_files']) * 100
            print(f"✓ Processing completed with {success_rate:.1f}% success rate!")
            print(f"📊 Detailed report available at: {report_path}")
        else:
            print("✗ No files were successfully processed.")
            sys.exit(1)

    except AudioProcessorError as e:
        logger.error(f"Audio processing error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

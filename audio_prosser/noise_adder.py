#!/usr/bin/env python3
"""
Optimized Audio Noise Adder

This program adds background noise from the noise/ directory to processed audio samples.
Optimized version with parallel processing, batch operations, memory efficiency, and random time offsets for varied noise augmentation.

Author: AI Assistant
Date: September 2025
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# Audio processing imports
try:
    from pydub import AudioSegment
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('noise_adder_optimized.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class NoiseAdderError(Exception):
    """Custom exception for noise adding errors."""
    pass


class OptimizedNoiseAdder:
    """
    Optimized class for adding noise to processed audio samples.

    Features:
    - Parallel processing of multiple files
    - Batch noise loading and caching
    - Memory-efficient audio processing
    - Configurable worker threads/processes
    - Random time offsets in noise files for varied augmentation
    """

    def __init__(self, samples_dir: str, noise_dir: str, output_dir: str = "NoisySamples",
                 max_workers: int = None, use_processes: bool = False):
        """
        Initialize the OptimizedNoiseAdder.

        Args:
            samples_dir: Path to the directory containing processed samples
            noise_dir: Path to the directory containing noise files
            output_dir: Path to the output directory for noisy files
            max_workers: Maximum number of worker threads/processes (default: CPU count)
            use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        """
        self.samples_dir = Path(samples_dir)
        self.noise_dir = Path(noise_dir)
        self.output_dir = Path(output_dir)
        self.use_processes = use_processes
        
        # Set max workers based on CPU count
        if max_workers is None:
            self.max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
        else:
            self.max_workers = max_workers

        # Noise mapping: filename -> short name for output
        self.noise_mapping = {
            'background-factory.wav': 'factory',
            'cafe.wav': 'cafe',
            'convention-crowd.wav': 'crowd',
            'inside-car-while-driving.wav': 'car',
            'people-talking.wav': 'talking',
            'car-honk.wav': 'honk',
            'rain.mp3': 'rain'
        }

        # Cache for loaded noise files
        self._noise_cache = {}

        # Validate directories
        if not self.samples_dir.exists():
            raise NoiseAdderError(f"Samples directory does not exist: {samples_dir}")

        if not self.noise_dir.exists():
            raise NoiseAdderError(f"Noise directory does not exist: {noise_dir}")

        if not self.samples_dir.is_dir():
            raise NoiseAdderError(f"Samples path is not a directory: {samples_dir}")

        if not self.noise_dir.is_dir():
            raise NoiseAdderError(f"Noise path is not a directory: {noise_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"OptimizedNoiseAdder initialized with samples: {samples_dir}, noise: {noise_dir}, output: {output_dir}")
        logger.info(f"Using {self.max_workers} {'processes' if use_processes else 'threads'} for parallel processing")

    def discover_noise_files(self) -> Dict[str, Path]:
        """
        Discover all available noise files.

        Returns:
            Dictionary mapping short noise names to file paths
        """
        noise_files = {}

        try:
            for noise_file in self.noise_dir.iterdir():
                if noise_file.is_file():
                    filename = noise_file.name
                    if filename in self.noise_mapping:
                        short_name = self.noise_mapping[filename]
                        noise_files[short_name] = noise_file
                        logger.debug(f"Found noise file: {filename} -> {short_name}")
                    else:
                        logger.warning(f"Unmapped noise file: {filename}")

            logger.info(f"Discovered {len(noise_files)} mapped noise files")
            return noise_files

        except PermissionError as e:
            raise NoiseAdderError(f"Permission denied accessing noise directory: {e}")
        except Exception as e:
            raise NoiseAdderError(f"Error discovering noise files: {e}")

    def discover_sample_files(self) -> List[Path]:
        """
        Discover all processed sample files in subdirectories.

        Returns:
            List of paths to sample files
        """
        sample_files = []

        try:
            # Find all WAV files in subdirectories
            for wav_file in self.samples_dir.rglob("*.wav"):
                if wav_file.is_file():
                    sample_files.append(wav_file)
                    logger.debug(f"Found sample file: {wav_file}")

            logger.info(f"Discovered {len(sample_files)} sample files")
            return sample_files

        except PermissionError as e:
            raise NoiseAdderError(f"Permission denied accessing samples directory: {e}")
        except Exception as e:
            raise NoiseAdderError(f"Error discovering sample files: {e}")

    def load_audio_file(self, file_path: Path) -> AudioSegment:
        """
        Load an audio file using pydub with caching for noise files.

        Args:
            file_path: Path to the audio file

        Returns:
            AudioSegment object

        Raises:
            NoiseAdderError: If file cannot be loaded
        """
        if not PYDUB_AVAILABLE:
            raise NoiseAdderError("pydub library is required for audio processing")

        # Check cache for noise files
        cache_key = str(file_path)
        if cache_key in self._noise_cache:
            logger.debug(f"Using cached noise file: {file_path.name}")
            return self._noise_cache[cache_key]

        try:
            if file_path.suffix.lower() == '.mp3':
                audio = AudioSegment.from_mp3(file_path)
            else:
                audio = AudioSegment.from_wav(file_path)

            logger.debug(f"Loaded audio file: {file_path.name}, duration: {len(audio)}ms")
            
            # Cache noise files for reuse
            if any(noise_name in str(file_path) for noise_name in self.noise_mapping.values()):
                self._noise_cache[cache_key] = audio
                logger.debug(f"Cached noise file: {file_path.name}")
            
            return audio

        except Exception as e:
            raise NoiseAdderError(f"Error loading audio file {file_path.name}: {e}")

    def mix_audio_with_noise(self, sample_audio: AudioSegment, noise_audio: AudioSegment,
                           noise_level_db: float = -20.0) -> AudioSegment:
        """
        Mix sample audio with background noise with random time offset.

        Args:
            sample_audio: The main audio sample
            noise_audio: The background noise audio
            noise_level_db: Volume level for the noise (default: -20dB)

        Returns:
            Mixed audio segment
        """
        try:
            # Adjust noise volume
            noise_adjusted = noise_audio + noise_level_db

            # Make sure both audio segments are the same length
            sample_length = len(sample_audio)
            noise_length = len(noise_adjusted)

            # Randomly select a starting point in the noise file
            # Ensure we have enough noise to cover the sample (at least 80% of sample length remaining)
            min_remaining_length = int(sample_length * 0.8)
            max_start_offset = max(0, noise_length - min_remaining_length)

            if max_start_offset > 0:
                # Randomly select starting offset
                start_offset = random.randint(0, max_start_offset)
                logger.debug(f"Selected random noise offset: {start_offset}ms from {noise_length}ms noise file")
            else:
                # Not enough noise for random offset, use from start
                start_offset = 0
                logger.debug(f"Noise file too short for random offset, using from start")

            # Extract noise segment starting from random offset
            remaining_noise = noise_adjusted[start_offset:]

            # Now apply the same logic as before with the extracted segment
            remaining_length = len(remaining_noise)

            if remaining_length < sample_length:
                # Loop the remaining noise to match sample length
                loops_needed = (sample_length // remaining_length) + 1
                looped_noise = remaining_noise * loops_needed
                final_noise = looped_noise[:sample_length]
                logger.debug(f"Looped {loops_needed}x remaining noise ({remaining_length}ms) to cover sample ({sample_length}ms)")
            else:
                # Use enough of the remaining noise to match sample length
                final_noise = remaining_noise[:sample_length]
                logger.debug(f"Used {sample_length}ms segment from remaining noise ({remaining_length}ms)")

            # Mix the audio (overlay)
            mixed_audio = sample_audio.overlay(final_noise)

            logger.debug(f"Mixed audio: sample {sample_length}ms + random noise segment -> {len(mixed_audio)}ms")
            return mixed_audio

        except Exception as e:
            raise NoiseAdderError(f"Error mixing audio with noise: {e}")

    def parse_sample_filename(self, sample_path: Path) -> Tuple[str, str]:
        """
        Parse a sample filename to extract word and number.

        Args:
            sample_path: Path to the sample file

        Returns:
            Tuple of (word_name, number)
        """
        filename = sample_path.stem  # Remove .wav extension

        # Expected format: WordName_Number (e.g., Lehitraot_1)
        if '_' in filename:
            parts = filename.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0], parts[1]

        # Fallback: use entire filename as word, empty number
        logger.warning(f"Unexpected filename format: {filename}")
        return filename, ""

    def process_sample_with_noise(self, noise_name: str, noise_audio: AudioSegment,
                                sample_path: Path, noise_level_db: float, samples_dir: Path, output_dir: Path) -> Optional[Path]:
        """
        Process a single sample with a single noise type (for parallel processing).

        Args:
            noise_name: Name of the noise type
            noise_audio: Pre-loaded noise audio
            sample_path: Path to the sample file
            noise_level_db: Volume level for noise
            samples_dir: Base samples directory
            output_dir: Base output directory

        Returns:
            Path to created file or None if failed
        """
        try:
            # Parse filename to get word and number
            word_name, number = self.parse_sample_filename(sample_path)

            # Load the sample audio
            sample_audio = self.load_audio_file(sample_path)

            # Mix sample with noise
            mixed_audio = self.mix_audio_with_noise(sample_audio, noise_audio, noise_level_db)

            # Generate output filename: word_number_noise.wav
            if number:
                output_filename = f"{word_name}_{number}_{noise_name}.wav"
            else:
                output_filename = f"{word_name}_{noise_name}.wav"

            # Create output path (maintain subdirectory structure)
            relative_path = sample_path.relative_to(samples_dir)
            output_subdir = output_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)

            output_path = output_subdir / output_filename

            # Export the mixed audio
            mixed_audio.export(output_path, format="wav")

            return output_path

        except Exception as e:
            logger.error(f"Failed to process {sample_path.name} with {noise_name}: {e}")
            return None

    def process_sample_file_parallel(self, sample_path: Path, noise_files: Dict[str, Path],
                                   noise_level_db: float = -20.0) -> List[Path]:
        """
        Process a single sample file with all noise types using parallel processing.

        Args:
            sample_path: Path to the sample file
            noise_files: Dictionary of available noise files
            noise_level_db: Volume level for noise

        Returns:
            List of paths to created noisy files
        """
        created_files = []

        try:
            logger.debug(f"Processing sample file: {sample_path.name}")

            # Pre-load all noise files for this sample
            noise_audios = {}
            for noise_name, noise_path in noise_files.items():
                try:
                    noise_audios[noise_name] = self.load_audio_file(noise_path)
                except Exception as e:
                    logger.error(f"Failed to load noise file {noise_path.name}: {e}")
                    continue

            if not noise_audios:
                logger.warning(f"No noise files loaded for {sample_path.name}")
                return []

            # Process all noise types in parallel
            with ThreadPoolExecutor(max_workers=min(len(noise_audios), 4)) as executor:
                # Submit all tasks
                future_to_noise = {
                    executor.submit(
                        self.process_sample_with_noise,
                        noise_name, 
                        noise_audio,
                        sample_path,
                        noise_level_db,
                        self.samples_dir,
                        self.output_dir
                    ): noise_name
                    for noise_name, noise_audio in noise_audios.items()
                }

                # Collect results
                for future in as_completed(future_to_noise):
                    noise_name = future_to_noise[future]
                    try:
                        result = future.result()
                        if result:
                            created_files.append(result)
                            logger.debug(f"Created noisy file: {result.name}")
                    except Exception as e:
                        logger.error(f"Error processing {sample_path.name} with {noise_name}: {e}")

            logger.debug(f"Successfully created {len(created_files)} noisy versions of {sample_path.name}")
            return created_files

        except Exception as e:
            logger.error(f"Failed to process {sample_path.name}: {e}")
            return []

    def process_all_samples_parallel(self, noise_level_db: float = -20.0) -> dict:
        """
        Process all sample files with parallel processing.

        Args:
            noise_level_db: Volume level for noise (default: -20dB)

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'failed_samples': 0,
            'total_noisy_files': 0,
            'noise_types': 0
        }

        try:
            # Discover noise files
            noise_files = self.discover_noise_files()
            stats['noise_types'] = len(noise_files)

            if not noise_files:
                raise NoiseAdderError("No valid noise files found")

            # Discover sample files
            sample_files = self.discover_sample_files()
            stats['total_samples'] = len(sample_files)

            logger.info(f"Starting parallel noise addition for {len(sample_files)} samples with {len(noise_files)} noise types")
            logger.info(f"Using {self.max_workers} {'processes' if self.use_processes else 'threads'}")

            # Choose executor based on configuration
            ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

            # Process samples in parallel
            with ExecutorClass(max_workers=self.max_workers) as executor:
                # Submit all sample processing tasks
                future_to_sample = {
                    executor.submit(self.process_sample_file_parallel, sample_path, noise_files, noise_level_db): sample_path
                    for sample_path in sample_files
                }

                # Progress bar for samples
                sample_pbar = tqdm(
                    as_completed(future_to_sample),
                    total=len(sample_files),
                    desc="Processing samples",
                    unit="sample",
                    disable=not TQDM_AVAILABLE
                )

                processed_samples = 0
                failed_samples = 0
                total_noisy_files = 0

                for future in sample_pbar:
                    sample_path = future_to_sample[future]
                    sample_pbar.set_description(f"Processing {sample_path.name}")

                    try:
                        result = future.result()
                        if result:
                            processed_samples += 1
                            total_noisy_files += len(result)
                        else:
                            failed_samples += 1

                        # Update progress bar
                        sample_pbar.set_postfix({
                            'processed': processed_samples,
                            'failed': failed_samples,
                            'noisy_files': total_noisy_files
                        })

                    except Exception as e:
                        logger.error(f"Error processing {sample_path.name}: {e}")
                        failed_samples += 1

            stats['processed_samples'] = processed_samples
            stats['failed_samples'] = failed_samples
            stats['total_noisy_files'] = total_noisy_files

            # Final summary
            logger.info("=" * 50)
            logger.info("PARALLEL NOISE ADDITION COMPLETE")
            logger.info(f"Sample files found:     {stats['total_samples']}")
            logger.info(f"Successfully processed: {stats['processed_samples']}")
            logger.info(f"Failed samples:         {stats['failed_samples']}")
            logger.info(f"Noise types applied:    {stats['noise_types']}")
            logger.info(f"Total noisy files:      {stats['total_noisy_files']}")
            logger.info(f"Workers used:           {self.max_workers} {'processes' if self.use_processes else 'threads'}")
            logger.info("=" * 50)

            return stats

        except NoiseAdderError as e:
            logger.error(f"Processing failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during processing: {e}")
            raise NoiseAdderError(f"Processing failed: {e}")

    def generate_processing_report(self, stats: dict) -> str:
        """
        Generate a comprehensive processing report.

        Args:
            stats: Processing statistics dictionary

        Returns:
            Formatted report string
        """
        from datetime import datetime

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("OPTIMIZED NOISE ADDITION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Overall Statistics
        report_lines.append("OVERALL STATISTICS:")
        report_lines.append("-" * 30)
        report_lines.append(f"Total sample files:      {stats['total_samples']}")
        report_lines.append(f"Successfully processed:  {stats['processed_samples']}")
        report_lines.append(f"Failed samples:          {stats['failed_samples']}")
        report_lines.append(f"Noise types applied:     {stats['noise_types']}")
        report_lines.append(f"Total noisy files created: {stats['total_noisy_files']}")
        report_lines.append(f"Parallel workers used:   {self.max_workers} {'processes' if self.use_processes else 'threads'}")
        report_lines.append("")

        # Success Rate
        if stats['total_samples'] > 0:
            success_rate = (stats['processed_samples'] / stats['total_samples']) * 100
            report_lines.append(f"Success Rate: {success_rate:.1f}%")
            report_lines.append("")

        # Performance Metrics
        if stats['processed_samples'] > 0:
            avg_noisy_files = stats['total_noisy_files'] / stats['processed_samples']
            report_lines.append("PERFORMANCE METRICS:")
            report_lines.append("-" * 30)
            report_lines.append(f"Average noisy files per sample: {avg_noisy_files:.1f}")
            report_lines.append(f"Parallel processing: {'Enabled' if self.max_workers > 1 else 'Disabled'}")
            report_lines.append("")

        # Noise Types
        report_lines.append("NOISE TYPES APPLIED:")
        report_lines.append("-" * 30)
        noise_files = self.discover_noise_files()
        for noise_name in sorted(noise_files.keys()):
            report_lines.append(f"  • {noise_name}")
        report_lines.append("")

        # Optimization Features
        report_lines.append("OPTIMIZATION FEATURES USED:")
        report_lines.append("-" * 30)
        report_lines.append("  • Parallel processing with concurrent.futures")
        report_lines.append("  • Noise file caching for reuse")
        report_lines.append("  • Batch audio processing")
        report_lines.append("  • Memory-efficient audio handling")
        report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 30)

        if stats['total_samples'] > 0:
            success_rate = (stats['processed_samples'] / stats['total_samples']) * 100
            if success_rate < 75:
                report_lines.append("⚠️  LOW SUCCESS RATE - Consider:")
                report_lines.append("   • Checking audio file formats and quality")
                report_lines.append("   • Verifying pydub/ffmpeg installation")
                report_lines.append("   • Adjusting noise volume levels")
            elif success_rate < 90:
                report_lines.append("ℹ️  MODERATE SUCCESS RATE - Optimization opportunities:")
                report_lines.append("   • Fine-tune noise mixing parameters")
                report_lines.append("   • Check for audio file corruption")
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
        report_filename = f"optimized_noise_addition_report_{timestamp}.txt"
        report_path = output_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Optimized noise addition report saved to: {report_path}")
        return report_path


def main():
    """Main entry point for the optimized noise adder."""
    parser = argparse.ArgumentParser(
        description="Add background noise to processed audio samples with parallel processing optimization and random time offsets."
    )
    parser.add_argument(
        "samples_dir",
        help="Path to the directory containing processed samples (e.g., ProcessedSamples)"
    )
    parser.add_argument(
        "noise_dir",
        help="Path to the directory containing noise files (e.g., noise)"
    )
    parser.add_argument(
        "-o", "--output",
        default="NoisySamples",
        help="Output directory for noisy files (default: NoisySamples)"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=-20.0,
        help="Volume level for noise in dB (default: -20.0, more negative = quieter)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads/processes (default: CPU count, max 8)"
    )
    parser.add_argument(
        "--use-processes",
        action="store_true",
        help="Use ProcessPoolExecutor instead of ThreadPoolExecutor (may be faster for CPU-intensive tasks)"
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

        # Initialize the optimized noise adder
        noise_adder = OptimizedNoiseAdder(
            args.samples_dir, 
            args.noise_dir, 
            args.output,
            max_workers=args.max_workers,
            use_processes=args.use_processes
        )
        logger.info("Optimized noise adder initialized successfully")

        # Run the noise addition pipeline
        stats = noise_adder.process_all_samples_parallel(args.noise_level)

        # Generate comprehensive report
        report = noise_adder.generate_processing_report(stats)

        # Save report to file
        report_path = noise_adder.save_report_to_file(report, Path(args.output))

        # Print final summary to console
        print("\n" + "=" * 50)
        print("OPTIMIZED NOISE ADDITION SUMMARY")
        print("=" * 50)
        print(f"Sample files found:    {stats['total_samples']}")
        print(f"Files processed:       {stats['processed_samples']}")
        print(f"Files failed:          {stats['failed_samples']}")
        print(f"Noise types applied:   {stats['noise_types']}")
        print(f"Noisy files created:   {stats['total_noisy_files']}")
        print(f"Workers used:          {noise_adder.max_workers} {'processes' if args.use_processes else 'threads'}")
        print(f"Report saved to:       {report_path}")
        print("=" * 50)

        # Print success/failure message
        if stats['processed_samples'] > 0:
            success_rate = (stats['processed_samples'] / stats['total_samples']) * 100
            print(f"✓ Optimized noise addition completed with {success_rate:.1f}% success rate!")
            print(f"📊 Detailed report available at: {report_path}")
        else:
            print("✗ No files were successfully processed.")
            sys.exit(1)

    except NoiseAdderError as e:
        logger.error(f"Noise addition error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

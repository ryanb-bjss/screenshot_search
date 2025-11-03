#!/usr/bin/env python3
"""
Screenshot Search CLI - Natural language search for screenshots
"""
import argparse
import sys
from pathlib import Path
from indexer import index_screenshots
from searcher import search_screenshots


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Screenshot Search - Natural language search for screenshots using OCR and visual descriptions"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Index command
    index_parser = subparsers.add_parser(
        'index',
        help='Index screenshots from a directory'
    )
    index_parser.add_argument(
        'screenshots_dir',
        type=str,
        help='Directory containing screenshots to index'
    )
    index_parser.add_argument(
        '--output-dir',
        type=str,
        default='index',
        help='Directory to save index (default: index)'
    )

    # Search command
    search_parser = subparsers.add_parser(
        'search',
        help='Search indexed screenshots'
    )
    search_parser.add_argument(
        'query',
        type=str,
        help='Natural language search query'
    )
    search_parser.add_argument(
        '--index-dir',
        type=str,
        default='index',
        help='Directory containing the index (default: index)'
    )
    search_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'index':
        # Validate directory exists
        if not Path(args.screenshots_dir).exists():
            print(f"Error: Directory '{args.screenshots_dir}' does not exist")
            sys.exit(1)

        print("Starting screenshot indexing...")
        index_screenshots(args.screenshots_dir, args.output_dir)

    elif args.command == 'search':
        # Validate index exists
        if not Path(args.index_dir).exists():
            print(f"Error: Index directory '{args.index_dir}' does not exist")
            print("Please run 'index' command first to create an index")
            sys.exit(1)

        search_screenshots(args.query, args.index_dir, args.top_k)


if __name__ == '__main__':
    main()

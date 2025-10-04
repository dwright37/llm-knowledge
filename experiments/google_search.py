import requests
from bs4 import BeautifulSoup
from googlesearch import search
import time
import random
import argparse
import json
import csv
import os
import re

# List of domains to ignore by default
DEFAULT_IGNORED_DOMAINS = [
    'facebook.com', 'fb.com',
    'x.com', 'twitter.com',
    'linkedin.com',
    'youtube.com',
    'bsky.app', 'bluesky.app',
    'vimeo.com',
    'instagram.com'
]

def google_search(query, num_results=20, min_chars=1000, ignored_domains=None, verbose=True):
    """
    Perform a Google search, extract content from each URL, and return results with sufficient content.

    Args:
        query (str): The search term to query Google with
        num_results (int): Number of results to return
        min_chars (int): Minimum number of characters required to include a result
        ignored_domains (list): List of domain names to ignore (default: None, which uses DEFAULT_IGNORED_DOMAINS)
        verbose (bool): Whether to print progress messages (default: True)

    Returns:
        list: A list of dictionaries containing URL, text content, and length
    """
    if verbose:
        print(f"Searching Google for: {query}")
    results = []
    urls_processed = 0

    # Use default ignored domains if none provided
    if ignored_domains is None:
        ignored_domains = DEFAULT_IGNORED_DOMAINS

    try:
        # Use the googlesearch library to get search results
        for url in search(query, num_results=num_results*2, unique=True, region="us"):  # Get more URLs to account for filtering
            urls_processed += 1
            if verbose:
                print(f"Found URL ({urls_processed}): {url}")

            # Check if URL is from an ignored domain
            if any(domain in url.lower() for domain in ignored_domains):
                if verbose:
                    print(f"✗ Skipped: URL from ignored domain: {url}")
                continue

            # Skip PDF files
            if url.lower().endswith('.pdf'):
                if verbose:
                    print(f"✗ Skipped: PDF file: {url}")
                continue

            # Extract text immediately
            if verbose:
                print(f"Extracting content from: {url}")
            text = extract_text_from_url(url, verbose=verbose)

            # Check if the content meets the minimum character requirement
            if text and len(text) >= min_chars:
                results.append({
                    'url': url,
                    'text': text,
                    'length': len(text)
                })
                if verbose:
                    print(f"✓ Added to results ({len(text)} characters)")
            else:
                if verbose:
                    if text:
                        print(f"✗ Skipped: Content too short ({len(text)} characters, minimum is {min_chars})")
                    else:
                        print(f"✗ Skipped: Failed to extract content")

            # Add a small delay to avoid being blocked
            time.sleep(random.uniform(1.0, 3.0))

            # Check if we have enough results
            if len(results) >= num_results:
                break

            # Limit the total number of URLs we process to avoid excessive runtime
            if urls_processed >= num_results * 3:
                if verbose:
                    print(f"Reached maximum URL processing limit ({urls_processed})")
                break

    except Exception as e:
        if verbose:
            print(f"Error during Google search: {e}")

    if verbose:
        print(f"\nProcessed {urls_processed} URLs, found {len(results)} with sufficient content")
    return results

def extract_text_from_url(url, verbose=True):
    """
    Download a webpage and extract its text content.

    Args:
        url (str): The URL to download and parse
        verbose (bool): Whether to print progress messages (default: True)

    Returns:
        str: The extracted text content
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Download the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())

        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        if verbose:
            print(f"Error extracting text from {url}: {e}")
        return ""

def process_queries_from_file(query_file, num_results=20, min_chars=1000, ignored_domains=None, verbose=True):
    """
    Process multiple search queries from a file and return results for all queries.

    Args:
        query_file (str): Path to a file containing search queries (one per line)
        num_results (int): Number of results to return per query
        min_chars (int): Minimum number of characters required to include a result
        ignored_domains (list): List of domain names to ignore
        verbose (bool): Whether to print progress messages

    Returns:
        dict: A dictionary with queries as keys and search results as values
    """
    all_results = {}

    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]

        if verbose:
            print(f"Found {len(queries)} queries in {query_file}")

        for i, query in enumerate(queries):
            if verbose:
                print(f"\n[{i+1}/{len(queries)}] Processing query: {query}")

            # Perform the search for this query
            results = google_search(query, num_results, min_chars, ignored_domains, verbose)

            # Store the results for this query
            all_results[query] = results

            # Add a delay between queries to avoid being blocked
            if i < len(queries) - 1:  # Don't delay after the last query
                delay = random.uniform(2.0, 5.0)
                if verbose:
                    print(f"Waiting {delay:.1f} seconds before next query...")
                time.sleep(delay)

        return all_results

    except Exception as e:
        if verbose:
            print(f"Error processing queries from file: {e}")
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Search Google and download text from result pages')
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--query', type=str, help='Search term to query Google with')
    query_group.add_argument('--query-file', type=str, help='File containing search queries (one per line)')

    parser.add_argument('--results', type=int, default=20, help='Number of results to retrieve per query (default: 20)')
    parser.add_argument('--min-chars', type=int, default=1000, help='Minimum characters required for a result (default: 1000)')
    parser.add_argument('--output', type=str, default='search_results.json', help='Output file name (default: search_results.json)')
    parser.add_argument('--format', type=str, choices=['txt', 'json', 'csv'], 
                        help='Output format (default: determined by file extension)')
    parser.add_argument('--ignore-domains', type=str, nargs='+', 
                        help='Additional domain names to ignore (e.g., example.com)')
    parser.add_argument('--no-default-ignore', action='store_true', 
                        help='Do not use the default list of ignored domains')

    args = parser.parse_args()

    # Determine output format based on file extension or format argument
    output_format = args.format
    if not output_format:
        _, ext = os.path.splitext(args.output)
        if ext:
            output_format = ext[1:].lower()  # Remove the dot and convert to lowercase
        else:
            output_format = 'json'  # Default format is now JSON

    # When using query-file, force JSON format
    if args.query_file and output_format != 'json':
        print("Warning: When using --query-file, output format must be JSON. Forcing JSON format.")
        output_format = 'json'

    # Prepare the list of domains to ignore
    ignored_domains = None
    if args.no_default_ignore:
        ignored_domains = args.ignore_domains or []
    elif args.ignore_domains:
        ignored_domains = DEFAULT_IGNORED_DOMAINS + args.ignore_domains

    if args.query:
        # Single query mode
        results = google_search(args.query, args.results, args.min_chars, ignored_domains)
        save_results(results, args.output, output_format, args.query, verbose=True)
    else:
        # Multiple queries from file mode
        all_results = process_queries_from_file(args.query_file, args.results, args.min_chars, ignored_domains)

        # Save all results to a single JSON file
        with open(args.output, 'w', encoding='utf-8') as f:
            json_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'queries': {}
            }

            # Add each query's results to the JSON structure
            for query, results in all_results.items():
                json_data['queries'][query] = results

            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Completed! Results for {len(all_results)} queries saved to {args.output} in JSON format")

def save_results(results, output_file, output_format='txt', query='', verbose=True):
    """
    Save search results to a file in the specified format.

    Args:
        results (list): List of dictionaries containing search results
        output_file (str): Path to the output file
        output_format (str): Format to save the results in ('txt', 'json', or 'csv')
        query (str): The search query that produced these results
        verbose (bool): Whether to print progress messages (default: True)

    Returns:
        None
    """
    if verbose:
        print(f"\nSaving {len(results)} results to {output_file}...")

    # Determine output format based on file extension if not specified
    if not output_format:
        _, ext = os.path.splitext(output_file)
        if ext:
            output_format = ext[1:].lower()  # Remove the dot and convert to lowercase
        else:
            output_format = 'txt'  # Default format

    # Save results in the specified format
    if output_format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json_data = {
                'query': query,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'results': results
            }
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    elif output_format == 'csv':
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Query', 'URL', 'Text Length', 'Text'])
            for result in results:
                writer.writerow([
                    query,
                    result['url'],
                    result['length'],
                    result['text'][:1000] + ('...' if len(result['text']) > 1000 else '')
                ])

    else:  # Default to txt format
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Search results for: {query}\n\n")
            for result in results:
                f.write(f"--- Content from {result['url']} ---\n\n{result['text']}\n\n")

    if verbose:
        print(f"Completed! Results saved to {output_file} in {output_format.upper()} format")

if __name__ == "__main__":
    main()

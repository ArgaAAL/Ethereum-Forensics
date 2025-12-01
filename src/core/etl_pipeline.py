# --- ethereum_csv_processor.py ---

import requests
import pandas as pd
import numpy as np
import json
import time
import os
from collections import defaultdict, Counter
from typing import List, Dict, Optional
from datetime import datetime, timezone

# Configuration
ETHERSCAN_API_KEY = "YOUR_ETHERSCAN_KEY_HERE"
CRYPTOCOMPARE_API_KEY = "YOUR_CRYPTOCOMPARE_KEY_HERE"
WEI_TO_ETH = 10**18
SATOSHI_TO_BTC = 10**8
API_DELAY = 0.2
MAX_RETRIES = 3
ETHERSCAN_MAX_RECORDS = 10000  # Etherscan's limit per request
MAX_TRANSACTIONS_PER_ADDRESS = 50000  # Cap total transactions per address

class TimeAwarePriceConverter:
    """
    Converts ETH to BTC using historical prices, but optimizes by fetching
    only one price per month to save on API calls.
    """
    
    def __init__(self):
        self.price_cache = {}
        self.load_price_cache()
    
    def load_price_cache(self):
        if os.path.exists('price_cache.json'):
            try:
                with open('price_cache.json', 'r') as f:
                    self.price_cache = json.load(f)
            except:
                self.price_cache = {}
    
    def save_price_cache(self):
        with open('price_cache.json', 'w') as f:
            json.dump(self.price_cache, f)
    
    def get_eth_btc_ratio(self, timestamp: int) -> float:
        """
        Get ETH/BTC ratio. Uses the first day of the month of the timestamp
        as the key to reduce API calls.
        """
        dt_object = datetime.fromtimestamp(timestamp)
        
        # --- KEY CHANGE HERE ---
        # Instead of the exact day, we use the 1st of the month as the key.
        # Example: 2018-10-23 becomes 2018-10-01
        # This means all transactions in October 2018 will share ONE API call.
        monthly_key = dt_object.strftime('%Y-%m-01')
        
        if monthly_key in self.price_cache:
            # Return the saved price for that month
            return self.price_cache[monthly_key]
        
        # --- NEW: Logic to get price from CryptoCompare ---
        try:
            # We need a timestamp for the first day of that month for the API call
            monthly_dt = datetime.strptime(monthly_key, '%Y-%m-%d')
            monthly_timestamp = int(monthly_dt.timestamp())

            print(f"-> No cached price for {monthly_key}. Fetching from API...")

            url = "https://min-api.cryptocompare.com/data/pricehistorical"
            params = {
                'fsym': 'ETH',
                'tsyms': 'BTC',
                'ts': monthly_timestamp,
                'api_key': CRYPTOCOMPARE_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # The response is nested under 'ETH'
                eth_price = data.get('ETH', {}).get('BTC')

                if eth_price:
                    self.price_cache[monthly_key] = eth_price
                    time.sleep(API_DELAY) # Wait a moment before the next potential call
                    return eth_price
                else:
                    # API call succeeded but didn't return a valid price
                    print(f"   Warning: Valid response but no price data for {monthly_key}. Using fallback.")
                    return self._get_fallback_ratio(timestamp)
            else:
                print(f"   Warning: API error (status {response.status_code}) for {monthly_key}. Using fallback.")
                return self._get_fallback_ratio(timestamp)

        except Exception as e:
            print(f"   Error: Price fetch failed for {monthly_key}: {e}. Using fallback.")
            return self._get_fallback_ratio(timestamp)
    
    def _get_fallback_ratio(self, timestamp: int) -> float:
        """Approximate ETH/BTC ratio based on time periods (unchanged)"""
        year = datetime.fromtimestamp(timestamp).year
        
        if year <= 2016: return 0.02
        elif year <= 2017: return 0.05
        elif year <= 2018: return 0.08
        elif year <= 2020: return 0.04
        elif year <= 2021: return 0.06
        else: return 0.067

class EthereumDataExtractor:
    """Extract Ethereum and ERC-20 data with proper fee calculation and transaction limit"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.price_converter = TimeAwarePriceConverter()
    
    def get_all_transactions(self, address: str) -> List[Dict]:
        """Get both ETH and ERC-20 transactions, merging them with transaction limit."""
        print(f"  Fetching ETH transactions for {address}...")
        eth_txs = self._get_eth_transactions(address)
        print(f"  Found {len(eth_txs)} ETH transactions")
        
        print(f"  Fetching ERC-20 transactions for {address}...")
        erc20_txs = self._get_erc20_transactions(address)
        print(f"  Found {len(erc20_txs)} ERC-20 transactions")

        # Hashes of transactions that are parents to an ERC-20 transfer
        erc20_parent_hashes = {tx['hash'] for tx in erc20_txs}
        eth_tx_map = {tx['hash']: tx for tx in eth_txs}
        
        all_txs = []

        # 1. Process ERC-20 transactions first, giving them priority
        for tx in erc20_txs:
            tx['tx_type'] = 'ERC20'
            tx_hash = tx.get('hash')
            if tx_hash in eth_tx_map:
                parent_tx = eth_tx_map[tx_hash]
                tx['gasUsed'] = parent_tx.get('gasUsed', '21000')  # Minimum gas
                tx['gasPrice'] = parent_tx.get('gasPrice', '20000000000')  # 20 gwei default
            all_txs.append(tx)

        # 2. Process only PURE ETH transactions (those not associated with an ERC-20 transfer)
        for tx in eth_txs:
            # If the hash was already processed as part of an ERC-20 transfer, skip it.
            if tx['hash'] in erc20_parent_hashes:
                continue
            
            # This is a pure ETH transfer (or a contract call with no tokens)
            tx['tx_type'] = 'ETH'
            all_txs.append(tx)
            
        # Sort by timestamp and apply transaction limit
        all_txs.sort(key=lambda x: int(x.get('timeStamp', 0)))
        
        # Apply transaction limit
        if len(all_txs) > MAX_TRANSACTIONS_PER_ADDRESS:
            print(f"  ⚠️  Limiting to {MAX_TRANSACTIONS_PER_ADDRESS} transactions (found {len(all_txs)})")
            all_txs = all_txs[:MAX_TRANSACTIONS_PER_ADDRESS]
        
        print(f"  Total combined transactions: {len(all_txs)}")
        return all_txs
    
    def _get_eth_transactions(self, address: str) -> List[Dict]:
        """Get ETH transactions with limit"""
        all_results = []
        start_block = 0
        page_count = 0
        
        while len(all_results) < MAX_TRANSACTIONS_PER_ADDRESS:
            page_count += 1
            print(f"    Fetching ETH page {page_count} (starting from block {start_block})...")
            
            page_results = self._fetch_eth_page(address, start_block)
            if not page_results:
                break
            
            all_results.extend(page_results)
            
            # Check if we've reached our limit
            if len(all_results) >= MAX_TRANSACTIONS_PER_ADDRESS:
                all_results = all_results[:MAX_TRANSACTIONS_PER_ADDRESS]
                print(f"    ⚠️  Reached ETH transaction limit of {MAX_TRANSACTIONS_PER_ADDRESS}")
                break
            
            # If we got less than the max records, we've reached the end
            if len(page_results) < ETHERSCAN_MAX_RECORDS:
                break
            
            # Set the next start_block to be one after the last one received
            last_block = int(page_results[-1]['blockNumber'])
            start_block = last_block + 1
            
        print(f"    Retrieved {len(all_results)} total ETH transactions across {page_count} pages")
        return all_results
    
    def _fetch_eth_page(self, address: str, start_block: int) -> List[Dict]:
        """Fetch a single page of ETH transactions"""
        for attempt in range(MAX_RETRIES):
            try:
                url = "https://api.etherscan.io/api"
                params = {
                    'module': 'account',
                    'action': 'txlist',
                    'address': address,
                    'startblock': start_block,
                    'endblock': 99999999,
                    'sort': 'asc',
                    'apikey': self.api_key
                }
                
                response = requests.get(url, params=params, timeout=30)
                data = response.json()
                
                if data.get('status') == '1':
                    time.sleep(API_DELAY)
                    return data.get('result', [])
                elif data.get('status') == '0' and 'No transactions found' in data.get('message', ''):
                    # No more transactions
                    return []
                else:
                    print(f"    ETH API error: {data.get('message', 'Unknown error')}")
                    if attempt == MAX_RETRIES - 1:
                        return []
                    
            except requests.exceptions.RequestException as e:
                print(f"    ETH fetch attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    return []
                time.sleep(2 ** attempt)
        
        return []
    
    def _get_erc20_transactions(self, address: str) -> List[Dict]:
        """Get ERC-20 transactions with limit"""
        all_results = []
        start_block = 0
        page_count = 0
        
        while len(all_results) < MAX_TRANSACTIONS_PER_ADDRESS:
            page_count += 1
            print(f"    Fetching ERC-20 page {page_count} (starting from block {start_block})...")
            
            page_results = self._fetch_erc20_page(address, start_block)
            if not page_results:
                break
            
            all_results.extend(page_results)
            
            # Check if we've reached our limit
            if len(all_results) >= MAX_TRANSACTIONS_PER_ADDRESS:
                all_results = all_results[:MAX_TRANSACTIONS_PER_ADDRESS]
                print(f"    ⚠️  Reached ERC-20 transaction limit of {MAX_TRANSACTIONS_PER_ADDRESS}")
                break
            
            # If we got less than the max records, we've reached the end
            if len(page_results) < ETHERSCAN_MAX_RECORDS:
                break
            
            # Set the next start_block to be one after the last one received
            last_block = int(page_results[-1]['blockNumber'])
            start_block = last_block + 1
            
        print(f"    Retrieved {len(all_results)} total ERC-20 transactions across {page_count} pages")
        return all_results
    
    def _fetch_erc20_page(self, address: str, start_block: int) -> List[Dict]:
        """Fetch a single page of ERC-20 transactions"""
        for attempt in range(MAX_RETRIES):
            try:
                url = "https://api.etherscan.io/api"
                params = {
                    'module': 'account',
                    'action': 'tokentx',
                    'address': address,
                    'startblock': start_block,
                    'endblock': 99999999,
                    'sort': 'asc',
                    'apikey': self.api_key
                }
                
                response = requests.get(url, params=params, timeout=30)
                data = response.json()
                
                if data.get('status') == '1':
                    time.sleep(API_DELAY)
                    return data.get('result', [])
                elif data.get('status') == '0' and 'No transactions found' in data.get('message', ''):
                    # No more transactions
                    return []
                else:
                    print(f"    ERC-20 API error: {data.get('message', 'Unknown error')}")
                    if attempt == MAX_RETRIES - 1:
                        return []
                    
            except requests.exceptions.RequestException as e:
                print(f"    ERC-20 fetch attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    return []
                time.sleep(2 ** attempt)
        
        return []

class FeatureCalculator:
    """Calculate Bitcoin-compatible features from Ethereum data"""
    
    def __init__(self, price_converter: TimeAwarePriceConverter):
        self.price_converter = price_converter
    
    def calculate_features(self, address: str, transactions: List[Dict]) -> Dict:
        """Calculate all features matching Bitcoin dataset schema"""
        if not transactions:
            return self._get_empty_features()
        
        address = address.lower()
        
        sent_txs = []
        received_txs = []
        all_values_satoshi = []
        all_fees_satoshi = []
        blocks = []
        counterparties = Counter()
        
        for tx in transactions:
            timestamp = int(tx.get('timeStamp', 0))
            block_num = int(tx.get('blockNumber', 0))
            tx_from = tx.get('from', '').lower()
            tx_to = tx.get('to', '').lower()
            
            # Convert value based on transaction type
            if tx.get('tx_type') == 'ETH':
                value_eth = int(tx.get('value', 0)) / WEI_TO_ETH
            else:  # ERC20
                decimals = int(tx.get('tokenDecimal', 18))
                value_eth = int(tx.get('value', 0)) / (10 ** decimals)
            
            # Always calculate gas fee (for both ETH and ERC-20)
            gas_used = int(tx.get('gasUsed', 0))
            gas_price = int(tx.get('gasPrice', 0))
            gas_fee_eth = (gas_used * gas_price) / WEI_TO_ETH
            
            # Convert to Satoshi-equivalent floats (CHANGED FROM int() TO float())
            if timestamp > 0:
                eth_btc_ratio = self.price_converter.get_eth_btc_ratio(timestamp)
                # Convert to Satoshi-equivalent floats to match Bitcoin dataset format
                value_satoshi = float((value_eth * eth_btc_ratio) * SATOSHI_TO_BTC)
                fee_satoshi = float((gas_fee_eth * eth_btc_ratio) * SATOSHI_TO_BTC)
            else:
                value_satoshi = fee_satoshi = 0.0
            
            blocks.append(block_num)
            
            # Determine if sent or received
            if tx_from == address:  # This is an outgoing transaction
                # ALWAYS append the fee for any outgoing transaction
                all_fees_satoshi.append(fee_satoshi)
                
                # Only count it as a value-sending transaction if value > 0
                if value_satoshi > 0:
                    sent_txs.append({
                        'value_satoshi': value_satoshi,
                        'fee_satoshi': fee_satoshi,
                        'block': block_num,
                        'timestamp': timestamp
                    })
                    all_values_satoshi.append(value_satoshi)
                    if tx_to:
                        counterparties[tx_to] += 1

            if tx_to == address:  # This is an incoming transaction
                if value_satoshi > 0:
                    received_txs.append({
                        'value_satoshi': value_satoshi,
                        'block': block_num,
                        'timestamp': timestamp
                    })
                    all_values_satoshi.append(value_satoshi)
                    if tx_from:
                        counterparties[tx_from] += 1
        
        # Calculate features
        features = {}
        
        # Basic counts (CHANGED TO float() FOR CONSISTENCY)
        features['num_txs_as_sender'] = float(len(sent_txs))
        features['num_txs_as receiver'] = float(len(received_txs))
        features['total_txs'] = float(len(transactions))
        
        # Block features (CHANGED TO float() FOR CONSISTENCY)
        if blocks:
            blocks = [b for b in blocks if b > 0]
            features['first_block_appeared_in'] = float(min(blocks)) if blocks else 0.0
            features['last_block_appeared_in'] = float(max(blocks)) if blocks else 0.0
            features['lifetime_in_blocks'] = float(max(blocks) - min(blocks)) if len(blocks) > 1 else 0.0
            features['num_timesteps_appeared_in'] = float(len(set(blocks)))
        else:
            features.update({
                'first_block_appeared_in': 0.0,
                'last_block_appeared_in': 0.0,
                'lifetime_in_blocks': 0.0,
                'num_timesteps_appeared_in': 0.0
            })
        
        # First transaction blocks (CHANGED TO float() FOR CONSISTENCY)
        sent_blocks = [tx['block'] for tx in sent_txs if tx['block'] > 0]
        received_blocks = [tx['block'] for tx in received_txs if tx['block'] > 0]
        features['first_sent_block'] = float(min(sent_blocks)) if sent_blocks else 0.0
        features['first_received_block'] = float(min(received_blocks)) if received_blocks else 0.0
        
        # Value statistics (now in Satoshi-equivalent floats)
        sent_values = [tx['value_satoshi'] for tx in sent_txs]
        received_values = [tx['value_satoshi'] for tx in received_txs]
        
        self._add_stats(features, 'btc_transacted', all_values_satoshi)
        self._add_stats(features, 'btc_sent', sent_values)
        self._add_stats(features, 'btc_received', received_values)
        
        # Fee statistics (now in Satoshi-equivalent floats)
        fees = [tx['fee_satoshi'] for tx in sent_txs]
        self._add_stats(features, 'fees', all_fees_satoshi)
        
        # Fee as share of value
        fee_shares = []
        for tx in sent_txs:
            if tx['value_satoshi'] > 0:
                share = (tx['fee_satoshi'] / tx['value_satoshi']) * 100
                fee_shares.append(share)
        self._add_stats(features, 'fees_as_share', fee_shares)
        
        # Block intervals (ENSURE RETURNED AS FLOATS)
        if len(blocks) > 1:
            sorted_blocks = sorted(set(blocks))
            intervals = [float(sorted_blocks[i] - sorted_blocks[i-1]) for i in range(1, len(sorted_blocks))]
            self._add_stats(features, 'blocks_btwn_txs', intervals)
        else:
            self._add_stats(features, 'blocks_btwn_txs', [])
        
        # Sent block intervals (ENSURE RETURNED AS FLOATS)
        if len(sent_blocks) > 1:
            sent_intervals = [float(sent_blocks[i] - sent_blocks[i-1]) for i in range(1, len(sent_blocks))]
            self._add_stats(features, 'blocks_btwn_input_txs', sent_intervals)
        else:
            self._add_stats(features, 'blocks_btwn_input_txs', [])
        
        # Received block intervals (ENSURE RETURNED AS FLOATS)
        if len(received_blocks) > 1:
            received_intervals = [float(received_blocks[i] - received_blocks[i-1]) for i in range(1, len(received_blocks))]
            self._add_stats(features, 'blocks_btwn_output_txs', received_intervals)
        else:
            self._add_stats(features, 'blocks_btwn_output_txs', [])
        
        # Address interaction features (ENSURE RETURNED AS FLOATS)
        unique_addresses = len(counterparties)
        multiple_interactions = len([addr for addr, count in counterparties.items() if count > 1])
        interaction_counts = list(counterparties.values())
        
        features['transacted_w_address_total'] = float(unique_addresses)
        features['num_addr_transacted_multiple'] = float(multiple_interactions)
        self._add_stats(features, 'transacted_w_address', interaction_counts, include_total=False)
        
        # Time step (use unique blocks as proxy) (ENSURE RETURNED AS FLOAT)
        features['Time step'] = float(features['num_timesteps_appeared_in'])
        
        return features
    
    def _add_stats(self, features: Dict, prefix: str, values: List[float], include_total: bool = True):
        """Add statistical aggregations for a list of values - ENSURE ALL RETURNS ARE FLOATS"""
        if values:
            # Convert all values to floats to ensure consistency
            float_values = [float(v) for v in values]
            
            if include_total:
                features[f'{prefix}_total'] = float(sum(float_values))
            features[f'{prefix}_min'] = float(min(float_values))
            features[f'{prefix}_max'] = float(max(float_values))
            features[f'{prefix}_mean'] = float(np.mean(float_values))
            features[f'{prefix}_median'] = float(np.median(float_values))
        else:
            if include_total:
                features[f'{prefix}_total'] = 0.0
            features[f'{prefix}_min'] = 0.0
            features[f'{prefix}_max'] = 0.0
            features[f'{prefix}_mean'] = 0.0
            features[f'{prefix}_median'] = 0.0
    
    def _get_empty_features(self) -> Dict:
        """Return empty feature set with all zeros AS FLOATS"""
        feature_names = [
            'num_txs_as_sender', 'num_txs_as receiver', 'first_block_appeared_in',
            'last_block_appeared_in', 'lifetime_in_blocks', 'total_txs',
            'first_sent_block', 'first_received_block', 'num_timesteps_appeared_in',
            'btc_transacted_total', 'btc_transacted_min', 'btc_transacted_max',
            'btc_transacted_mean', 'btc_transacted_median', 'btc_sent_total',
            'btc_sent_min', 'btc_sent_max', 'btc_sent_mean', 'btc_sent_median',
            'btc_received_total', 'btc_received_min', 'btc_received_max',
            'btc_received_mean', 'btc_received_median', 'fees_total', 'fees_min',
            'fees_max', 'fees_mean', 'fees_median', 'fees_as_share_total',
            'fees_as_share_min', 'fees_as_share_max', 'fees_as_share_mean',
            'fees_as_share_median', 'blocks_btwn_txs_total', 'blocks_btwn_txs_min',
            'blocks_btwn_txs_max', 'blocks_btwn_txs_mean', 'blocks_btwn_txs_median',
            'blocks_btwn_input_txs_total', 'blocks_btwn_input_txs_min',
            'blocks_btwn_input_txs_max', 'blocks_btwn_input_txs_mean',
            'blocks_btwn_input_txs_median', 'blocks_btwn_output_txs_total',
            'blocks_btwn_output_txs_min', 'blocks_btwn_output_txs_max',
            'blocks_btwn_output_txs_mean', 'blocks_btwn_output_txs_median',
            'num_addr_transacted_multiple', 'transacted_w_address_total',
            'transacted_w_address_min', 'transacted_w_address_max',
            'transacted_w_address_mean', 'transacted_w_address_median', 'Time step'
        ]
        
        # ALL ZEROS AS FLOATS
        return {name: 0.0 for name in feature_names}

class EthereumProcessor:
    """Main processor for Ethereum addresses"""
    
    def __init__(self, api_key: str = ETHERSCAN_API_KEY):
        self.extractor = EthereumDataExtractor(api_key)
        self.calculator = FeatureCalculator(self.extractor.price_converter)
    
    def process_address(self, address: str) -> Optional[Dict]:
        """Process single Ethereum address"""
        try:
            print(f"Processing {address}...")
            
            transactions = self.extractor.get_all_transactions(address)
            if not transactions:
                print(f"No transactions found for {address}")
                return None
            
            features = self.calculator.calculate_features(address, transactions)
            features['address'] = address
            
            print(f"✅ {address}: {len(transactions)} transactions processed")
            return features
            
        except Exception as e:
            print(f"❌ Error processing {address}: {e}")
            return None
    
    def process_batch(self, addresses: List[str], output_file: str = 'ethereum_features.csv') -> pd.DataFrame:
        """Process multiple addresses"""
        results = []
        
        for address in addresses:
            result = self.process_address(address)
            if result:
                results.append(result)
        
        if results:
            df = pd.DataFrame(results)
            
            # DATA TYPE CLEANUP - ENSURE PROPER TYPES TO MATCH BITCOIN DATASET
            df = self._clean_data_types(df)
            
            df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to {output_file}")
            
            # Save price cache
            self.extractor.price_converter.save_price_cache()
            
            return df
        else:
            print("❌ No addresses processed successfully")
            return pd.DataFrame()

    def process_from_csv(self, csv_file_path: str, output_file: str = 'ethereum_features_with_labels.csv') -> pd.DataFrame:
        """
        Process addresses from the downloaded CSV dataset with periodic saving and resume capability.
        """
        try:
            print(f"📖 Reading source dataset from {csv_file_path}...")
            df_source = pd.read_csv(csv_file_path)
            addresses_with_labels = [
                (row['Address'], row['FLAG']) for _, row in df_source.iterrows()
            ]
            print(f"Found {len(addresses_with_labels)} total addresses in source file.")

        except Exception as e:
            print(f"❌ Error reading source CSV file: {e}")
            return pd.DataFrame()

        results = []
        processed_addresses = set()
        
        # --- Resume Logic ---
        if os.path.exists(output_file):
            try:
                print(f"📖 Found existing output file. Loading previous results from {output_file}...")
                df_existing = pd.read_csv(output_file)
                # Ensure the 'address' column exists before proceeding
                if 'address' in df_existing.columns:
                    results = df_existing.to_dict('records')
                    processed_addresses = set(df_existing['address'])
                    print(f"✅ Resuming. {len(processed_addresses)} addresses already processed.")
                else:
                    print(f"⚠️ Existing output file '{output_file}' is malformed and missing 'address' column. Starting fresh.")
            except pd.errors.EmptyDataError:
                print(f"⚠️ Existing output file '{output_file}' is empty. Starting fresh.")
            except Exception as e:
                 print(f"⚠️ Could not read existing output file. Starting fresh. Error: {e}")

        
        processed_in_session = 0
        total_to_process = len(addresses_with_labels)

        for i, (address, flag) in enumerate(addresses_with_labels, 1):
            # Skip if address has already been processed in a previous session
            if address in processed_addresses:
                continue

            print(f"\n--- Processing {i}/{total_to_process}: {address} ---")
            
            result = self.process_address(address)
            
            if result:
                result['class'] = flag  # Add the class label
                results.append(result)
                processed_in_session += 1
                
                # --- Periodic Save Logic ---
                if processed_in_session > 0 and processed_in_session % 10 == 0:
                    print(f"\n💾 Saving progress... 10 new addresses processed in this session.")
                    df_current_results = pd.DataFrame(results)
                    df_current_results = self._clean_data_types(df_current_results)
                    df_current_results.to_csv(output_file, index=False)
                    self.extractor.price_converter.save_price_cache()
                    print(f"   Total of {len(results)} addresses now saved to {output_file}.")
            else:
                print(f"⚠️  Skipping {address} - no transaction data could be retrieved.")
        
        # --- Final Save ---
        if results:
            print("\n✅ Processing complete. Performing final save...")
            df_final = pd.DataFrame(results)
            df_final = self._clean_data_types(df_final)
            df_final.to_csv(output_file, index=False)
            self.extractor.price_converter.save_price_cache()
            print(f"💾 All results saved to {output_file}. Total processed addresses: {len(df_final)}")
            return df_final
        else:
            print("❌ No new addresses were processed in this session.")
            return pd.DataFrame()
    
    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data types to match Bitcoin dataset format"""
        # Ensure all numeric columns are proper floats
        numeric_columns = [col for col in df.columns if col not in ['address', 'class']]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        return df

# Usage example
if __name__ == "__main__":
    processor = EthereumProcessor()
    
    # Use the CSV dataset you downloaded
    # IMPORTANT: Make sure to use the correct path for your system.
    # Using a raw string (r"...") on Windows or a regular string on Mac/Linux is recommended.
    csv_path = r"D:\AAL\Coding\piton\BlockchainAnalyzer\transaction_dataset.csv\transaction_dataset.csv"
    
    # Process all addresses from the CSV with resume and periodic saving
    results_df = processor.process_from_csv(csv_path)
    
    if not results_df.empty:
        print(f"\nFinal result: {len(results_df)} addresses processed with features and labels.")
    else:
        print("\nNo data was processed.")
import requests
import time
import json
from datetime import datetime
from typing import Dict, Tuple

# --- CONSTANTS ---
# Replace with your actual API keys
ETHERSCAN_API_KEY = "YOUR_ETHERSCAN_KEY_HERE"
CRYPTOCOMPARE_API_KEY = "YOUR_CRYPTOCOMPARE_KEY_HERE"
# Get your free Moralis API key from https://moralis.io/
MORALIS_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6IjA0MWY5NzljLTY1MWEtNGRkOS1hZGNlLTE1ZWI1NzI2MTg5YiIsIm9yZ0lkIjoiNDU5ODc5IiwidXNlcklkIjoiNDczMTMxIiwidHlwZUlkIjoiM2RjNDVmNTEtNDJhNy00MDM0LThhZDMtZjQwYzc3ZjdkZjAzIiwidHlwZSI6IlBST0pFQ1QiLCJpYXQiOjE3NTI3NzIyMDcsImV4cCI6NDkwODUzMjIwN30.nEiBgcrCEQj1ST_J6yiLj9VsrR3kSHmzm8Ho5YmSdTw"
API_DELAY = 0.3 # Increased slightly for safety with multiple services

class ImprovedTimeAwarePriceConverter:
    """Enhanced version with better logging and verification"""

    def __init__(self, use_approximations: bool = True):
        self.price_cache = {}
        self.token_price_cache = {}
        self.token_info_cache = {}
        self.use_approximations = use_approximations
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.approximation_count = 0
        self.failed_api_count = 0

    def get_token_info(self, token_address: str) -> Dict[str, any]:
        """Gets token info from a hardcoded map or Etherscan."""
        token_address = token_address.lower()
        if token_address in self.token_info_cache:
            self.cache_hit_count += 1
            return self.token_info_cache[token_address]

        token_map = {
            '0xdac17f958d2ee523a2206206994597c13d831ec7': {'symbol': 'USDT', 'decimals': 6},
            '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48': {'symbol': 'USDC', 'decimals': 6},
            '0x6b175474e89094c44da98b954eedeac495271d0f': {'symbol': 'DAI', 'decimals': 18},
            '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2': {'symbol': 'WETH', 'decimals': 18},
            '0x4fabb145d64652a948d72533023f6e7a623c7c53': {'symbol': 'BUSD', 'decimals': 18},
            '0xd0a4b8946cb52f0661273bfbc6fd0e0c75fc6433': {'symbol': 'OMG', 'decimals': 18},
            '0x41e5560054824ea6b0732e656e3ad64e20e94e45': {'symbol': 'CVC', 'decimals': 8},
            '0x621d78f2ef2fd937bfca696cabaf9a779f59b3ed': {'symbol': 'NEWB', 'decimals': 0},
        }

        if token_address in token_map:
            token_info = token_map[token_address]
            print(f"   -> Found {token_info['symbol']} in hardcoded map.")
            self.token_info_cache[token_address] = token_info
            return token_info

        print(f"   -> Fetching token info for {token_address} from Etherscan API...")
        self.api_call_count += 1
        params = {'module': 'contract', 'action': 'getsourcecode', 'address': token_address, 'apikey': ETHERSCAN_API_KEY}
        try:
            response = requests.get("https://api.etherscan.io/api", params=params, timeout=15)
            time.sleep(API_DELAY)
            response.raise_for_status()
            data = response.json()
            if data['status'] == '1' and data['result']:
                symbol = data['result'][0].get('ContractName', 'UNKNOWN').upper()
                token_info = {'symbol': symbol, 'decimals': 18}
                print(f"   -> Successfully fetched: {symbol}")
            else:
                token_info = {'symbol': 'UNKNOWN', 'decimals': 18}
            self.token_info_cache[token_address] = token_info
            return token_info
        except Exception as e:
            print(f"   -> Error fetching token info: {e}")
            self.failed_api_count += 1
            return {'symbol': 'UNKNOWN', 'decimals': 18}

    def get_token_eth_ratio(self, token_address: str, timestamp: int) -> float:
        """Main function to get the Token/ETH price ratio."""
        token_info = self.get_token_info(token_address)
        token_symbol = token_info['symbol']

        if token_symbol == 'WETH':
            return 1.0

        dt_object = datetime.fromtimestamp(timestamp)
        monthly_key = dt_object.strftime('%Y-%m-01')
        cache_key = f"{token_symbol}_{monthly_key}"

        if cache_key in self.token_price_cache:
            self.cache_hit_count += 1
            return self.token_price_cache[cache_key]

        if token_symbol in ['USDT', 'USDC', 'DAI', 'BUSD']:
            ratio = self._get_stablecoin_eth_ratio(timestamp)
            if ratio > 0: self.token_price_cache[cache_key] = ratio
            return ratio

        if token_symbol == 'UNKNOWN':
            return self._get_unknown_token_price(token_address, timestamp, cache_key)

        print(f"   -> API FETCH: Attempting to get {token_symbol}/ETH price for {monthly_key}")
        return self._fetch_token_price_from_api(token_symbol, timestamp, cache_key, token_address)

    def _fetch_from_moralis(self, token_address: str, timestamp: int) -> float:
        """Fallback to fetch price data from Moralis API."""
        print("      -> Attempting fallback to Moralis...")
        try:
            url = f"https://deep-index.moralis.io/api/v2.2/erc20/{token_address}/price"
            params = {'chain': 'eth', 'to_date': datetime.fromtimestamp(timestamp).isoformat()}
            headers = {"accept": "application/json", "X-API-Key": MORALIS_API_KEY}

            self.api_call_count += 1
            response = requests.get(url, headers=headers, params=params, timeout=15)
            time.sleep(API_DELAY)

            if response.status_code == 200:
                data = response.json()
                if data.get('usdPrice'):
                    token_usd_price = data['usdPrice']
                    # Now, convert this USD price to ETH
                    # We get the 1 / (ETH price in USD) from our existing stablecoin function
                    eth_per_usd = self._get_stablecoin_eth_ratio(timestamp)
                    if eth_per_usd > 0:
                        ratio = token_usd_price * eth_per_usd
                        print(f"      -> ✅ MORALIS SUCCESS: Found price {ratio:.8f} ETH")
                        return ratio
            
            print(f"      -> ❌ MORALIS FAILED. Status: {response.status_code}, Response: {response.text}")
            if response.status_code != 200:
                 self.failed_api_count += 1
            return 0.0
        except Exception as e:
            print(f"      -> ❌ MORALIS EXCEPTION: {e}")
            self.failed_api_count += 1
            return 0.0

    def _fetch_token_price_from_api(self, token_symbol: str, timestamp: int, cache_key: str, token_address: str) -> float:
        """Fetch price from CryptoCompare, with Moralis as a fallback."""
        try:
            monthly_dt = datetime.fromtimestamp(timestamp).replace(day=1)
            monthly_timestamp = int(monthly_dt.timestamp())
            params = {'fsym': token_symbol, 'tsyms': 'ETH', 'ts': monthly_timestamp, 'api_key': CRYPTOCOMPARE_API_KEY}
            
            print(f"      -> Making API call to CryptoCompare for {token_symbol}...")
            self.api_call_count += 1
            response = requests.get("https://min-api.cryptocompare.com/data/pricehistorical", params=params, timeout=15)
            time.sleep(API_DELAY)

            if response.status_code == 200:
                data = response.json()
                token_price = data.get(token_symbol, {}).get('ETH')
                if token_price and token_price > 0:
                    print(f"      -> ✅ CryptoCompare SUCCESS: {token_symbol} = {token_price:.8f} ETH")
                    self.token_price_cache[cache_key] = token_price
                    return token_price
            print(f"      -> ❌ CryptoCompare FAILED: {response.json().get('Message', 'No data')}")
            self.failed_api_count += 1
        except Exception as e:
            print(f"      -> ❌ CryptoCompare EXCEPTION: {e}")
            self.failed_api_count += 1

        # --- Fallback to Moralis ---
        moralis_price = self._fetch_from_moralis(token_address, timestamp)
        if moralis_price > 0:
            self.token_price_cache[cache_key] = moralis_price
            return moralis_price

        # --- If all APIs fail, use approximation ---
        if self.use_approximations:
            print(f"      -> Using approximation as a last resort...")
            approx_ratio = self._get_token_price_approximation(token_symbol, timestamp)
            self.approximation_count += 1
            self.token_price_cache[cache_key] = approx_ratio
            return approx_ratio
        
        print(f"      -> ❌ ALL METHODS FAILED: {token_symbol} = 0.0 ETH")
        self.token_price_cache[cache_key] = 0.0
        return 0.0

    def _get_unknown_token_price(self, token_address: str, timestamp: int, cache_key: str) -> float:
        """Handles unknown tokens. Tries Moralis first, then approximation."""
        print(f"      -> Symbol for {token_address} is UNKNOWN. Trying Moralis directly.")
        moralis_price = self._fetch_from_moralis(token_address, timestamp)
        if moralis_price > 0:
            self.token_price_cache[cache_key] = moralis_price
            return moralis_price
        
        if self.use_approximations:
            print(f"      -> Using approximation for UNKNOWN token...")
            approx_ratio = self._get_token_price_approximation('UNKNOWN', timestamp)
            self.approximation_count += 1
            self.token_price_cache[cache_key] = approx_ratio
            return approx_ratio
        return 0.0

    def _get_stablecoin_eth_ratio(self, timestamp: int) -> float:
        """Gets 1 / (ETH price in USD), effectively the ETH per USD."""
        dt_object = datetime.fromtimestamp(timestamp)
        monthly_key = dt_object.strftime('%Y-%m-01')
        cache_key = f"ETH_USD_{monthly_key}"
        if cache_key in self.price_cache:
            self.cache_hit_count += 1
            eth_usd_price = self.price_cache[cache_key]
            return 1.0 / eth_usd_price if eth_usd_price > 0 else 0.0
        try:
            monthly_timestamp = int(datetime.strptime(monthly_key, '%Y-%m-%d').timestamp())
            params = {'fsym': 'ETH', 'tsyms': 'USD', 'ts': monthly_timestamp, 'api_key': CRYPTOCOMPARE_API_KEY}
            
            self.api_call_count += 1
            response = requests.get("https://min-api.cryptocompare.com/data/pricehistorical", params=params, timeout=15)
            time.sleep(API_DELAY)
            
            if response.status_code == 200:
                data = response.json()
                eth_usd_price = data.get('ETH', {}).get('USD')
                if eth_usd_price and eth_usd_price > 0:
                    self.price_cache[cache_key] = eth_usd_price
                    return 1.0 / eth_usd_price
            print(f"   Warning: Could not fetch ETH/USD price. Stablecoin value is zero.")
            return 0.0
        except Exception as e:
            print(f"   Error: ETH/USD fetch failed: {e}. Stablecoin value is zero.")
            return 0.0

    def _get_token_price_approximation(self, token_symbol: str, timestamp: int) -> float:
        """Provides a generic price approximation based on the year."""
        year = datetime.fromtimestamp(timestamp).year
        if year <= 2017: estimated_ratio = 0.0001
        elif year <= 2018: estimated_ratio = 0.00005
        else: estimated_ratio = 0.00001
        print(f"         -> 📊 GENERIC APPROXIMATION: {token_symbol} in {year} = {estimated_ratio:.8f} ETH")
        return estimated_ratio
    
    # ... (print_statistics and verify_suspicious_prices methods remain the same) ...
    def print_statistics(self):
        """Print detailed statistics about price fetching"""
        # Avoid division by zero if no requests were made
        total_requests = self.api_call_count + self.cache_hit_count + self.approximation_count
        if total_requests == 0:
            print("\nNo price requests were made.")
            return

        print(f"\n{'='*50}")
        print(f"PRICE FETCHING STATISTICS")
        print(f"{'='*50}")
        print(f"Total Price Lookups: {total_requests}")
        print(f"Cache Hits: {self.cache_hit_count} ({self.cache_hit_count/total_requests:.1%})")
        print(f"API Calls Made: {self.api_call_count}")
        print(f"  - Successful: {self.api_call_count - self.failed_api_count}")
        print(f"  - Failed: {self.failed_api_count}")
        print(f"Approximations Used: {self.approximation_count} ({self.approximation_count/total_requests:.1%})")
        print(f"{'='*50}")

    def verify_suspicious_prices(self):
        """Check for suspicious patterns in cached prices"""
        print(f"\n{'='*50}")
        print(f"SUSPICIOUS PRICE ANALYSIS")
        print(f"{'='*50}")

        price_frequency = {}
        for token_key, price in self.token_price_cache.items():
            price_frequency.setdefault(price, []).append(token_key)
        
        approximation_values = {0.0001, 0.00005, 0.00001}
        for price, tokens in price_frequency.items():
            if len(tokens) > 1 and price > 0:
                print(f"Price {price:.8f} ETH appears for {len(tokens)} tokens: {', '.join(tokens)}")
            if price in approximation_values:
                print(f"⚠️  Approximation value {price:.8f} ETH used for: {', '.join(tokens)}")

def test_price_fetching():
    """Test the price fetching with detailed logging"""
    converter = ImprovedTimeAwarePriceConverter(use_approximations=True)
    test_cases = [
        ('0xd0a4b8946cb52f0661273bfbc6fd0e0c75fc6433', 1499817600),  # July 2017 - OMG
        ('0x41e5560054824ea6b0732e656e3ad64e20e94e45', 1501459200),  # July 31, 2017 - CVC
        ('0x621d78f2ef2fd937bfca696cabaf9a779f59b3ed', 1499817600),  # July 2017 - NEWB
        ('0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48', 1609459200),  # Jan 2021 - USDC
    ]
    for token_address, timestamp in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing token: {token_address}")
        print(f"Timestamp: {timestamp} ({datetime.fromtimestamp(timestamp)})")
        print(f"{'='*60}")
        ratio = converter.get_token_eth_ratio(token_address, timestamp)
        print(f"\nFinal ratio for {token_address}: {ratio:.8f} ETH")

    converter.print_statistics()
    converter.verify_suspicious_prices()

if __name__ == "__main__":
    test_price_fetching()
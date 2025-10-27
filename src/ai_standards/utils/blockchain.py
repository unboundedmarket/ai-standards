"""
Blockchain integration utilities (mock and real implementations)
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path


class MockBlockchain:
    """
    Mock blockchain for testing and development.
    Stores records in memory and optionally persists to disk.
    """
    
    def __init__(self, persist_path: Optional[Path] = None):
        self.records: List[Dict[str, Any]] = []
        self.persist_path = persist_path
        
        if persist_path and persist_path.exists():
            self._load_records()
    
    def store_record(self, record: Dict[str, Any]) -> str:
        """
        Store a record on the mock blockchain.
        
        Args:
            record: Data to store
            
        Returns:
            Transaction ID (simulated)
        """
        tx_id = f"tx_{len(self.records)}_{datetime.utcnow().timestamp()}"
        
        entry = {
            "tx_id": tx_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": record,
        }
        
        self.records.append(entry)
        
        if self.persist_path:
            self._save_records()
        
        return tx_id
    
    def get_record(self, tx_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a record by transaction ID"""
        for entry in self.records:
            if entry["tx_id"] == tx_id:
                return entry["data"]
        return None
    
    def query_records(self, filter_fn=None) -> List[Dict[str, Any]]:
        """Query records with optional filter function"""
        if filter_fn is None:
            return [entry["data"] for entry in self.records]
        return [entry["data"] for entry in self.records if filter_fn(entry["data"])]
    
    def _save_records(self):
        """Persist records to disk"""
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'w') as f:
                json.dump(self.records, f, indent=2)
    
    def _load_records(self):
        """Load records from disk"""
        if self.persist_path and self.persist_path.exists():
            with open(self.persist_path, 'r') as f:
                self.records = json.load(f)


class BlockchainConnector:
    """
    Base class for blockchain connectors.
    Can be extended for actual Cardano/Ethereum integration.
    """
    
    def __init__(self, network: str = "testnet", use_mock: bool = True):
        self.network = network
        self.use_mock = use_mock
        
        if use_mock:
            self.backend = MockBlockchain()
        else:
            # TODO: Implement real blockchain connection
            raise NotImplementedError("Real blockchain connection not yet implemented")
    
    def mint_nft(self, metadata: Dict[str, Any]) -> str:
        """
        Mint an NFT with given metadata.
        
        Args:
            metadata: NFT metadata
            
        Returns:
            NFT token ID or transaction hash
        """
        record = {
            "type": "nft_mint",
            "metadata": metadata,
        }
        return self.backend.store_record(record)
    
    def store_benchmark(self, benchmark_data: Dict[str, Any]) -> str:
        """
        Store benchmark results on-chain.
        
        Args:
            benchmark_data: Benchmark results
            
        Returns:
            Transaction ID
        """
        record = {
            "type": "benchmark",
            "data": benchmark_data,
        }
        return self.backend.store_record(record)
    
    def get_model_card(self, nft_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model card by NFT ID"""
        record = self.backend.get_record(nft_id)
        if record and record.get("type") == "nft_mint":
            return record.get("metadata")
        return None
    
    def get_benchmark_results(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all benchmark results for a model"""
        return self.backend.query_records(
            lambda r: r.get("type") == "benchmark" and 
                     r.get("data", {}).get("model_id") == model_id
        )


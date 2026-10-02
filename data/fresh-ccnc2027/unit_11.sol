pragma solidity ^0.8.27;
contract Unit {
    uint256 public total;
    function price(uint256 quantity, uint256 unitPrice) external { unchecked { total = quantity * unitPrice; } }
}

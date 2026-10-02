pragma solidity ^0.8.27;
contract Unit {
    uint256 public remaining = 20;
    function reserve(uint256 amount) external { remaining -= amount; }
}

pragma solidity ^0.8.27;
contract Unit {
    uint8 public used;
    function add(uint8 amount) external { unchecked { used += amount; } }
}

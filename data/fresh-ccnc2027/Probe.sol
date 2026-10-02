pragma solidity ^0.8.27;
interface IUnit { function deposit() external payable; function withdraw() external; }
contract Probe {
    IUnit public unit; uint256 public rounds;
    constructor(address target) { unit = IUnit(target); }
    function begin() external payable { unit.deposit{value:msg.value}(); unit.withdraw(); }
    receive() external payable {
        if (rounds < 1) { rounds++; (bool ok,) = address(unit).call(abi.encodeWithSignature("withdraw()")); ok; }
    }
}
contract Rejector { receive() external payable { revert(); } }

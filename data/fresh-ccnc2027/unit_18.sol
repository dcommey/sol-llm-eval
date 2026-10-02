pragma solidity ^0.8.27;
contract Unit {
    bool public completed;
    function execute(address payable receiver) external payable {
        completed = true; bool ok = _deliver(receiver, msg.value); require(ok);
    }
    function _deliver(address payable receiver, uint256 value) internal returns (bool) { (bool ok,) = receiver.call{value: value}(""); return ok; }
}
